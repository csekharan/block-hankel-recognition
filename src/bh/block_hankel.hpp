#pragma once
#include <cstring>
#include <type_traits>
// Block-Hankel detection on a square m x m array A: all non-trivial admissible (p,q), i.e.
// p | m, q | m, p < m, q < m, (p,q) != (1,1)  (pairs with exactly one coordinate equal to 1 are
// tested), such that A is (p,q) block Hankel  <=>  (-p, q) is a quad-II period of A.
#include "../gp/galil_park.hpp"
#include "divisors.hpp"
#include <set>
#include <map>
#include <utility>

namespace bh {

using Pair = std::pair<int, int>;

struct Answer {
    std::set<Pair> pairs;      // non-trivial pairs
    std::set<Pair> trivial;    // p == m or q == m (single block row / column)
};

// Oracle: compare adjacent anti-diagonal blocks. O(m^2) per pair.
template <class T>
bool is_block_hankel_brute(const gp::Array2D<T>& A, int p, int q) {
    int m = A.n, M = m / p, N = m / q;
    for (int i = 1; i <= M - 1; ++i)
        for (int j = 0; j <= N - 2; ++j)
            for (int r = 0; r < p; ++r)
                for (int c = 0; c < q; ++c)
                    if (!(A.at(i * p + r, j * q + c) == A.at((i - 1) * p + r, (j + 1) * q + c))) return false;
    return true;
}

template <class T>
Answer find_pairs_brute(const gp::Array2D<T>& A) {
    Answer ans; int m = A.n;
    auto d = divisors(m);
    for (int p : d) for (int q : d) {
        if ((p == 1 && q == 1) || (p == m && q == m)) continue;
        if (p == m || q == m) { ans.trivial.insert({p, q}); continue; }
        if (is_block_hankel_brute(A, p, q)) ans.pairs.insert({p, q});
    }
    return ans;
}

// Row super-symbol Z-pass: for fixed q, all p in [1, m) with (-p, q) a period of A.
// Sequence right_0..right_{m-1} $ left_0..left_{m-1}, left_x = A[x, 0..m-q), right_x = A[x, q..m).
template <class T>
std::vector<char> zpass_rows(const gp::Array2D<T>& A, int q) {
    int m = A.n, L = m - q;
    auto rowptr = [&](int idx) { return idx < m ? &A.at(idx, q) : &A.at(idx - m - 1, 0); };
    auto eq = [&](int i, int j) {
        if (i == m || j == m) return false;
        const T* a = rowptr(i); const T* b = rowptr(j);
        if constexpr (std::is_trivially_copyable_v<T>) return std::memcmp(a, b, (std::size_t)L * sizeof(T)) == 0;   // same primitive as the direct baseline
        else { for (int k = 0; k < L; ++k) if (!(a[k] == b[k])) return false; return true; }
    };
    auto z = gp::zfunc(2 * m + 1, eq);
    std::vector<char> res(m, 0);
    for (int p = 1; p < m; ++p) res[p] = z[m + 1 + p] >= m - p;
    return res;
}

// Galil-Park front end: table lookup for short pairs, Z-passes for the long ones.
template <class T>
Answer find_pairs_gp(const gp::Array2D<T>& A, const gp::Result& R) {
    Answer ans; int m = A.n; int K = R.table.cap;
    auto d = divisors(m);
    std::vector<int> longs;
    for (int x : d) if (x > K && x != m) longs.push_back(x);     // m/2, m/3, m/4 at most
    // long q: rows of A;  long p: rows of A^T with q_B = p
    std::map<int, std::vector<char>> by_q, by_p;
    for (int q : longs) by_q[q] = zpass_rows(A, q);
    if (!longs.empty()) { auto B = A.transposed(); for (int p : longs) by_p[p] = zpass_rows(B, p); }
    for (int p : d) for (int q : d) {
        if ((p == 1 && q == 1) || (p == m && q == m)) continue;
        if (p == m || q == m) { ans.trivial.insert({p, q}); continue; }
        bool yes;
        if (p <= K && q <= K) yes = R.table.at({-p, q}).s == gp::Status::Period;
        else if (q > K) yes = by_q[q][p];
        else yes = by_p[p][q];
        if (yes) ans.pairs.insert({p, q});
    }
    return ans;
}

} // namespace bh
