#pragma once
// Monte Carlo block-Hankel recognizer, implemented exactly as in
//   Chandra N. Sekharan, "Efficient Detection of Nontrivial Block Hankel Structure: Algorithms and
//   Performance Comparison", manuscript, 2026 (the definitions, propositions, algorithm and theorems
//   of its polynomial-hashing sections; numbering below follows the earlier draft).
// Conventions: zero-based A[a:b, c:d]; T_{p,q} = A[0:m-p, q:n], B_{p,q} = A[p:m, 0:n-q];
// recthash(R) = sum A[i,j] beta1^{c2-1-j} beta2^{r2-1-i} (mod P), column base beta1, row base beta2;
// prefix table G[r,c] = recthash(0,0,r,c) built by the recurrences of Proposition 5; rectangle
// extraction by Proposition 6; k independent families (Definition 7) accept a pair only if all agree.
#include <cstdint>
#include <vector>
#include <random>
#include <utility>
#include <set>
#include <optional>
#include <cmath>
#include <algorithm>
#include <memory>
#include "divisors.hpp"

namespace bh {

// A rectangular matrix view: element (r, c) at data[r * stride + c].
template <class T>
struct RectView {
    const T* data; int rows, cols; std::size_t stride;
    const T& at(int r, int c) const { return data[(std::size_t)r * stride + c]; }
};

// Field F_P for a Mersenne-type prime P = 2^e - 1 (e = 61 uses the fast fold; others use % on 128 bits).
struct Field {
    std::uint64_t P;
    explicit Field(std::uint64_t P_) : P(P_) {}
    static constexpr std::uint64_t M61 = (1ULL << 61) - 1;
    // branchless conditional subtraction of P
    std::uint64_t red(std::uint64_t s) const { return s - (P & (0 - (std::uint64_t)(s >= P))); }
    std::uint64_t mul(std::uint64_t a, std::uint64_t b) const {
        unsigned __int128 x = (unsigned __int128)a * b;
        if (P == M61) {
            std::uint64_t lo = (std::uint64_t)x & M61, hi = (std::uint64_t)(x >> 61);
            return red(lo + hi);                        // lo + hi < 2^62, one reduction suffices
        }
        return (std::uint64_t)(x % P);
    }
    std::uint64_t add(std::uint64_t a, std::uint64_t b) const { return red(a + b); }
    std::uint64_t sub(std::uint64_t a, std::uint64_t b) const { return a >= b ? a - b : a + P - b; }
    // integer entry -> residue. Entries already in [0, P) (the usual case) cost one compare;
    // anything else takes the general path. Definition 2 assumes distinct entries stay distinct mod P.
    std::uint64_t val(long long x) const {
        std::uint64_t v = (std::uint64_t)x;
        if (v < P) return v;                            // 0 <= x < P
        long long r = x % (long long)P; if (r < 0) r += (long long)P; return (std::uint64_t)r;
    }
    std::uint64_t random_base(std::mt19937_64& rng) const {     // uniform in F_P^x = [1, P-1]
        std::uniform_int_distribution<std::uint64_t> d(1, P - 1); return d(rng);
    }
};

// One 2D polynomial-hash family: prefix table G plus power tables (Propositions 5, 6).
template <class T>
struct HashFamily {
    Field F; int m, n; std::uint64_t b1, b2;            // beta1 column base, beta2 row base
    std::unique_ptr<std::uint64_t[]> G;                 // (m+1) x (n+1), row 0 zero, rest filled by build
    std::vector<std::uint64_t> pw1, pw2;                // pw1[0..n], pw2[0..m]
    // allocate (uninitialized except row 0) and fill the power tables; G is filled by build_families
    HashFamily(int rows, int cols, Field F_, std::uint64_t beta1, std::uint64_t beta2)
        : F(F_), m(rows), n(cols), b1(beta1), b2(beta2),
          G(new std::uint64_t[(std::size_t)(rows + 1) * (cols + 1)]), pw1(cols + 1), pw2(rows + 1) {
        for (int c = 0; c <= n; ++c) G[c] = 0;
        pw1[0] = pw2[0] = 1;
        for (int c = 1; c <= n; ++c) pw1[c] = F.mul(pw1[c - 1], b1);
        for (int r = 1; r <= m; ++r) pw2[r] = F.mul(pw2[r - 1], b2);
    }
    HashFamily(const RectView<T>& A, Field F_, std::uint64_t beta1, std::uint64_t beta2)
        : HashFamily(A.rows, A.cols, F_, beta1, beta2) {
        HashFamily* self = this;
        build_families(A, std::vector<HashFamily*>{self});
    }
    std::uint64_t* row(int r) { return &G[(std::size_t)r * (n + 1)]; }
    std::uint64_t g(int r, int c) const { return G[(std::size_t)r * (n + 1) + c]; }
    // Proposition 6: hash of A[r1:r2, c1:c2]
    std::uint64_t recthash(int r1, int c1, int r2, int c2) const {
        std::uint64_t a = g(r2, c2);
        std::uint64_t bq = F.mul(g(r1, c2), pw2[r2 - r1]);
        std::uint64_t cq = F.mul(g(r2, c1), pw1[c2 - c1]);
        std::uint64_t dq = F.mul(F.mul(g(r1, c1), pw1[c2 - c1]), pw2[r2 - r1]);
        return F.add(F.sub(F.sub(a, bq), cq), dq);
    }
    std::uint64_t hT(int p, int q) const { return recthash(0, q, m - p, n); }   // T_{p,q}
    std::uint64_t hB(int p, int q) const { return recthash(p, 0, m, n - q); }   // B_{p,q}

    // Fill the G tables of several families in one pass over A (Proposition 5 recurrences).
    // Rows are processed in groups of RB so that the row-prefix Horner chains of RB rows and of
    // all K families run interleaved in registers: independent multiply chains fill the pipeline
    // instead of waiting on each other. The vertical recurrence is column-parallel already.
    template <int K>
    static void build_kernel(const RectView<T>& A, HashFamily* const* fams) {
        constexpr int RB = 4;
        const int m = A.rows, n = A.cols;
        const Field F = fams[0]->F;
        const std::uint64_t P = F.P;
        std::uint64_t b1[K]; for (int f = 0; f < K; ++f) b1[f] = fams[f]->b1;
        std::vector<std::uint64_t> Rbuf((std::size_t)K * RB * (n + 1));
        auto Rrow = [&](int f, int i) { return &Rbuf[((std::size_t)f * RB + i) * (n + 1)]; };
        for (int r0 = 0; r0 < m; r0 += RB) {
            const int rb = std::min(RB, m - r0);
            std::uint64_t acc[K][RB] = {};
            const T* Ar[RB]; std::uint64_t* Rp[K][RB];
            for (int i = 0; i < RB; ++i) Ar[i] = &A.at(r0 + std::min(i, rb - 1), 0);
            for (int f = 0; f < K; ++f) for (int i = 0; i < RB; ++i) { Rp[f][i] = Rrow(f, i); Rp[f][i][0] = 0; }
            if (P == Field::M61) {
                for (int c = 1; c <= n; ++c) {
                    std::uint64_t v[RB];
                    for (int i = 0; i < RB; ++i) v[i] = F.val((long long)Ar[i][c - 1]);
                    for (int f = 0; f < K; ++f)
                        for (int i = 0; i < RB; ++i) {
                            unsigned __int128 x = (unsigned __int128)acc[f][i] * b1[f];
                            std::uint64_t t = ((std::uint64_t)x & Field::M61) + (std::uint64_t)(x >> 61) + v[i];   // < 2^63
                            t = (t & Field::M61) + (t >> 61);
                            t -= Field::M61 & (0 - (std::uint64_t)(t >= Field::M61));
                            acc[f][i] = t; Rp[f][i][c] = t;
                        }
                }
            } else {
                for (int c = 1; c <= n; ++c)
                    for (int i = 0; i < RB; ++i) {
                        std::uint64_t vv = F.val((long long)Ar[i][c - 1]);
                        for (int f = 0; f < K; ++f) { acc[f][i] = F.add(F.mul(acc[f][i], b1[f]), vv); Rp[f][i][c] = acc[f][i]; }
                    }
            }
            for (int f = 0; f < K; ++f) {
                HashFamily& H = *fams[f];
                const std::uint64_t b2 = H.b2;
                for (int i = 0; i < rb; ++i) {
                    const std::uint64_t* Gp = H.row(r0 + i);
                    std::uint64_t* Gn = H.row(r0 + i + 1);
                    const std::uint64_t* Rr = Rp[f][i];
                    if (P == Field::M61) {
                        for (int c = 0; c <= n; ++c) {
                            unsigned __int128 x = (unsigned __int128)Gp[c] * b2;
                            std::uint64_t t = ((std::uint64_t)x & Field::M61) + (std::uint64_t)(x >> 61) + Rr[c];
                            t = (t & Field::M61) + (t >> 61);
                            t -= Field::M61 & (0 - (std::uint64_t)(t >= Field::M61));
                            Gn[c] = t;
                        }
                    } else {
                        for (int c = 0; c <= n; ++c) Gn[c] = F.add(F.mul(Gp[c], b2), Rr[c]);
                    }
                }
            }
        }
    }
    static void build_families(const RectView<T>& A, const std::vector<HashFamily*>& fams) {
        switch (fams.size()) {
            case 1: build_kernel<1>(A, fams.data()); break;
            case 2: build_kernel<2>(A, fams.data()); break;
            case 3: build_kernel<3>(A, fams.data()); break;
            case 4: build_kernel<4>(A, fams.data()); break;
            default: for (auto* f : fams) { HashFamily* one = f; build_kernel<1>(A, &one); } break;
        }
    }
};

// Build k families jointly (one pass over A).
template <class T>
std::vector<HashFamily<T>> build_families(const RectView<T>& A, Field F, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& bases) {
    std::vector<HashFamily<T>> fam;
    fam.reserve(bases.size());
    for (auto [b1, b2] : bases) fam.emplace_back(A.rows, A.cols, F, b1, b2);
    std::vector<HashFamily<T>*> ptrs; for (auto& f : fam) ptrs.push_back(&f);
    HashFamily<T>::build_families(A, ptrs);
    return fam;
}

struct PQ { int p, q; bool operator<(const PQ& o) const { return p != o.p ? p < o.p : q < o.q; } bool operator==(const PQ&) const = default; };

// Non-trivial admissible pairs, Step 4 convention (used throughout): p | m, q | n, p < m, q < n,
// (p,q) != (1,1). Pairs with exactly one coordinate equal to 1 are tested.
inline std::vector<PQ> candidate_pairs(int m, int n) {
    std::vector<PQ> out;
    for (int p : divisors(m)) for (int q : divisors(n)) {
        if (p >= m || q >= n) continue;
        if (p == 1 && q == 1) continue;
        out.push_back({p, q});
    }
    return out;
}

struct Recognition {
    std::vector<PQ> accepted;         // every tested pair on which all k families agree
    std::optional<PQ> first;          // Algorithm 1's output: the first accepted pair, or NONE
    double bound_exact = 0, bound_simple = 0;   // Theorem 1(iii) / Theorem 2 with the tested set
    std::vector<std::pair<std::uint64_t, std::uint64_t>> bases;
};

// k-fold recognizer (Definition 7). Bases are drawn from rng, uniformly and independently.
template <class T>
Recognition recognize(const RectView<T>& A, Field F, int k, std::mt19937_64& rng) {
    Recognition R;
    for (int r = 0; r < k; ++r) { std::uint64_t b1 = F.random_base(rng), b2 = F.random_base(rng); R.bases.push_back({b1, b2}); }
    std::vector<HashFamily<T>> fam = build_families(A, F, R.bases);
    int m = A.rows, n = A.cols;
    auto pairs = candidate_pairs(m, n);
    double dP = (double)(F.P - 1);
    for (PQ pq : pairs) {
        bool all = true;
        for (auto& f : fam) if (f.hT(pq.p, pq.q) != f.hB(pq.p, pq.q)) { all = false; break; }
        if (all) { R.accepted.push_back(pq); if (!R.first) R.first = pq; }
        double eps = ((double)(m - pq.p) + (double)(n - pq.q) - 2.0) / dP;
        R.bound_exact += std::pow(eps, k);
    }
    R.bound_simple = (double)pairs.size() * std::pow(((double)m + n - 4.0) / dP, k);
    return R;
}

// Deterministic oracle on a rectangular matrix (Definition 3 / Proposition 2), O(mn) per pair.
template <class T>
bool is_block_hankel_rect(const RectView<T>& A, int p, int q) {
    for (int i = 0; i < A.rows - p; ++i)
        for (int j = q; j < A.cols; ++j)
            if (!(A.at(i, j) == A.at(i + p, j - q))) return false;
    return true;
}

} // namespace bh
