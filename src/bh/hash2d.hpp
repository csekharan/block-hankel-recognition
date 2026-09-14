#pragma once
// 2-D polynomial hashing comparator: two independent hashes mod 2^61-1 with random bases.
#include "../gp/array2d.hpp"
#include "divisors.hpp"
#include "block_hankel.hpp"
#include <random>

namespace bh {

struct Mersenne61 {
    static constexpr std::uint64_t P = (1ULL << 61) - 1;
    static std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
        unsigned __int128 x = (unsigned __int128)a * b;
        std::uint64_t lo = (std::uint64_t)(x & P), hi = (std::uint64_t)(x >> 61);
        std::uint64_t s = lo + hi;
        if (s >= P) s -= P;
        return s;
    }
    static std::uint64_t add(std::uint64_t a, std::uint64_t b) { std::uint64_t s = a + b; if (s >= P) s -= P; return s; }
    static std::uint64_t sub(std::uint64_t a, std::uint64_t b) { return a >= b ? a - b : a + P - b; }
};

template <class T>
struct Hash2D {
    int m; std::uint64_t b1, b2;
    std::vector<std::uint64_t> H, pw1, pw2;   // (m+1)x(m+1) prefix table
    Hash2D(const gp::Array2D<T>& A, std::uint64_t base1, std::uint64_t base2)
        : m(A.n), b1(base1), b2(base2), H((std::size_t)(m + 1) * (m + 1), 0), pw1(m + 1), pw2(m + 1) {
        pw1[0] = pw2[0] = 1;
        for (int i = 1; i <= m; ++i) { pw1[i] = Mersenne61::mul(pw1[i - 1], b1); pw2[i] = Mersenne61::mul(pw2[i - 1], b2); }
        for (int x = 0; x < m; ++x) for (int y = 0; y < m; ++y) {
            std::uint64_t val = Mersenne61::mul(Mersenne61::mul(((std::uint64_t)A.at(x, y) + 1) % Mersenne61::P, pw1[x]), pw2[y]);
            std::uint64_t s = Mersenne61::add(val, Mersenne61::add(at(x, y + 1), at(x + 1, y)));
            at(x + 1, y + 1) = Mersenne61::sub(s, at(x, y));
        }
    }
    std::uint64_t& at(int x, int y) { return H[(std::size_t)x * (m + 1) + y]; }
    std::uint64_t rect(int x0, int y0, int x1, int y1) const {   // inclusive corners
        const auto& h = *this;
        auto g = [&](int x, int y) { return h.H[(std::size_t)x * (m + 1) + y]; };
        std::uint64_t s = Mersenne61::sub(Mersenne61::add(g(x1 + 1, y1 + 1), g(x0, y0)), Mersenne61::add(g(x0, y1 + 1), g(x1 + 1, y0)));
        return s;
    }
    // equality of A[x0..x0+h) x [y0..y0+w) and A[x0'..] x [y0'..]
    bool rect_equal(int x0, int y0, int x0p, int y0p, int h, int w) const {
        if (h <= 0 || w <= 0) return true;
        std::uint64_t r1 = rect(x0, y0, x0 + h - 1, y0 + w - 1), r2 = rect(x0p, y0p, x0p + h - 1, y0p + w - 1);
        std::uint64_t l = Mersenne61::mul(Mersenne61::mul(r1, pw1[x0p]), pw2[y0p]);
        std::uint64_t r = Mersenne61::mul(Mersenne61::mul(r2, pw1[x0]), pw2[y0]);
        return l == r;
    }
};

template <class T>
Answer find_pairs_hash(const gp::Array2D<T>& A, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint64_t> dist(2, Mersenne61::P - 2);
    Hash2D<T> h1(A, dist(rng), dist(rng)), h2(A, dist(rng), dist(rng));
    Answer ans; int m = A.n;
    auto d = divisors(m);
    for (int p : d) for (int q : d) {
        if ((p == 1 && q == 1) || (p == m && q == m)) continue;
        if (p == m || q == m) { ans.trivial.insert({p, q}); continue; }
        // (-p,q) period: A[p..m, 0..m-q) == A[0..m-p, q..m)
        bool yes = h1.rect_equal(p, 0, 0, q, m - p, m - q) && h2.rect_equal(p, 0, 0, q, m - p, m - q);
        if (yes) ans.pairs.insert({p, q});
    }
    return ans;
}

} // namespace bh
