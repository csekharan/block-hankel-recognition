#pragma once
// Input generators for tests. Each structured generator is itself validated against the
// brute-force classifier in the tests.
#include "../gp/galil_park.hpp"
#include <random>
#include <unordered_map>
#include <set>
#include <numeric>

namespace gen {
using gp::Array2D; using gp::Vec2;

inline Array2D<int> random_array(int m, int alpha, std::uint64_t seed) {
    std::mt19937_64 rng(seed); std::uniform_int_distribution<int> d(0, alpha - 1);
    Array2D<int> A(m);
    for (auto& x : A.a) x = d(rng);
    return A;
}

// Lattice-periodic: random symbol per class of the lattice (v1, v2).
inline Array2D<int> lattice(int m, Vec2 v1, Vec2 v2, int alpha, std::uint64_t seed) {
    std::mt19937_64 rng(seed); std::uniform_int_distribution<int> d(0, alpha - 1);
    gp::Lattice L(v1, v2);
    std::unordered_map<std::uint64_t, int> sym;
    Array2D<int> A(m);
    for (int r = 0; r < m; ++r) for (int c = 0; c < m; ++c) {
        auto k = L.key64({r, c});
        auto it = sym.find(k);
        if (it == sym.end()) it = sym.emplace(k, d(rng)).first;
        A.at(r, c) = it->second;
    }
    return A;
}

// Line-periodic along direction v: random symbol per residue class along v.
inline Array2D<int> line_dir(int m, Vec2 v, int alpha, std::uint64_t seed) {
    std::mt19937_64 rng(seed); std::uniform_int_distribution<int> d(0, alpha - 1);
    int g = std::gcd(gp::iabs(v.r), gp::iabs(v.c));
    Vec2 v0{v.r / g, v.c / g};
    auto fdiv = [](int a, int b) { int q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q; };
    std::unordered_map<long long, int> sym;
    Array2D<int> A(m);
    for (int r = 0; r < m; ++r) for (int c = 0; c < m; ++c) {
        Vec2 u{r, c};
        long cr = gp::cross(u, v0);
        int t = v0.r != 0 ? fdiv(u.r, v0.r) : fdiv(u.c, v0.c);
        int res = ((t % g) + g) % g;
        long long key = cr * 64 + res;
        auto it = sym.find(key);
        if (it == sym.end()) it = sym.emplace(key, d(rng)).first;
        A.at(r, c) = it->second;
    }
    return A;
}

// (p,q) block Hankel: random block sequence H_0..H_{M+N-2}, B_{i,j} = H_{i+j}.
inline Array2D<int> hankel_blocks(int m, int p, int q, int alpha, std::uint64_t seed) {
    std::mt19937_64 rng(seed); std::uniform_int_distribution<int> d(0, alpha - 1);
    int M = m / p, N = m / q;
    std::vector<std::vector<int>> H(M + N - 1, std::vector<int>(p * q));
    for (auto& blk : H) for (auto& x : blk) x = d(rng);
    Array2D<int> A(m);
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j)
        for (int r = 0; r < p; ++r) for (int c = 0; c < q; ++c)
            A.at(i * p + r, j * q + c) = H[i + j][r * q + c];
    return A;
}

} // namespace gen

// ---------------------------------------------------------------- Milestone 2 generators
namespace gen {

// Compare a computed table with the brute-force table over view V; witnesses are validated.
inline bool validate_table(const Array2D<int>& A, const gp::View& V, const gp::WitnessTable& got,
                           const gp::WitnessTable& want, const char* tag) {
    bool ok = true; int K = V.K();
    for (bool q1 : {true, false})
        for (Vec2 v : (q1 ? gp::WitnessTable::boxI(K) : gp::WitnessTable::boxII(K))) {
            const gp::Witness& g = got.at(v); const gp::Witness& w = want.at(v);
            if (g.s == gp::Status::Unknown) { std::fprintf(stderr, "%s: Unknown at (%d,%d)\n", tag, v.r, v.c); ok = false; continue; }
            if (g.s != w.s) { std::fprintf(stderr, "%s: status mismatch at (%d,%d): got %d want %d\n", tag, v.r, v.c, (int)g.s, (int)w.s); ok = false; continue; }
            if (g.s == gp::Status::Witness && !gp::check_witness(A, V, v, g.w)) {
                std::fprintf(stderr, "%s: invalid witness (%d,%d) for (%d,%d)\n", tag, g.w.r, g.w.c, v.r, v.c); ok = false;
            }
        }
    return ok;
}

// Top-level inner view (P^{t-1} of the final stage) for base size 16.
inline gp::View top_inner(int m, int base = 16) {
    gp::View V{0, 0, m};
    if (V.size <= base) return V;
    return V.inner();
}

// Lattice array with `count` random single-point defects planted outside the top inner view,
// restricted to the given absolute quadrants (bitmask bit k = quadrant k).
inline Array2D<int> lattice_with_point_defects(int m, Vec2 v1, Vec2 v2, int alpha, int count,
                                               unsigned quadmask, std::uint64_t seed) {
    Array2D<int> A = lattice(m, v1, v2, alpha, seed);
    std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ULL);
    gp::View V{0, 0, m}, inner = top_inner(m);
    int planted = 0, guard = 0;
    while (planted < count && guard++ < 100000) {
        int r = (int)(rng() % m), c = (int)(rng() % m);
        Vec2 p{r, c};
        if (inner.contains(p)) continue;
        bool okq = false; for (int k = 1; k <= 4; ++k) if ((quadmask >> k) & 1 && V.in_quadrant(p, k)) okq = true;
        if (!okq) continue;
        A.at(p) = alpha + 1 + (int)(rng() % 3);      // guaranteed to differ from lattice symbols
        ++planted;
    }
    return A;
}

// Lattice array with a random rectangular block of defects in one absolute quadrant.
inline Array2D<int> lattice_with_block_defects(int m, Vec2 v1, Vec2 v2, int alpha, int quadrant, std::uint64_t seed) {
    Array2D<int> A = lattice(m, v1, v2, alpha, seed);
    std::mt19937_64 rng(seed ^ 0x51ed27f3ULL);
    gp::View V{0, 0, m}, inner = top_inner(m);
    for (int r = 0; r < m; ++r) for (int c = 0; c < m; ++c) {
        Vec2 p{r, c};
        if (inner.contains(p) || !V.in_quadrant(p, quadrant)) continue;
        // a corner-anchored block of about a third of the quadrant
        int lr = r, lc = c; int h = V.half();
        bool in_block = (quadrant == 1) ? (lr < h / 3 && lc < h / 3)
                      : (quadrant == 2) ? (lr >= m - h / 3 && lc < h / 3)
                      : (quadrant == 3) ? (lr >= m - h / 3 && lc >= m - h / 3)
                                        : (lr < h / 3 && lc >= m - h / 3);
        if (in_block) A.at(p) = (int)(rng() % alpha);
    }
    return A;
}

// A3 vertical: basis (v1, (-k,0)); columns outside the top inner column band get a random
// vertically k-periodic pattern (defects w.r.t. the lattice, but (-k,0) stays a period).
inline Array2D<int> lattice_with_vertical_defects(int m, Vec2 v1, int k, int alpha, std::uint64_t seed) {
    Vec2 v2{-k, 0};
    Array2D<int> A = lattice(m, v1, v2, alpha, seed);
    std::mt19937_64 rng(seed ^ 0xabcdefULL);
    gp::View inner = top_inner(m);
    std::vector<std::vector<int>> g(m, std::vector<int>(k));
    for (auto& col : g) for (auto& x : col) x = (int)(rng() % alpha);
    for (int c = 0; c < m; ++c) {
        if (c >= inner.c0 && c < inner.c0 + inner.size) continue;
        for (int r = 0; r < m; ++r) A.at(r, c) = g[c][r % k];
    }
    return A;
}
// A3 horizontal: basis ((0,k), v2); rows outside the band get a random horizontally k-periodic pattern.
inline Array2D<int> lattice_with_horizontal_defects(int m, int k, Vec2 v2, int alpha, std::uint64_t seed) {
    Vec2 v1{0, k};
    Array2D<int> A = lattice(m, v1, v2, alpha, seed);
    std::mt19937_64 rng(seed ^ 0xfedcbaULL);
    gp::View inner = top_inner(m);
    std::vector<std::vector<int>> g(m, std::vector<int>(k));
    for (auto& row : g) for (auto& x : row) x = (int)(rng() % alpha);
    for (int r = 0; r < m; ++r) {
        if (r >= inner.r0 && r < inner.r0 + inner.size) continue;
        for (int c = 0; c < m; ++c) A.at(r, c) = g[r][c % k];
    }
    return A;
}

// Line overlay: residue classes along v1 whose line meets the top inner square keep the
// lattice symbols; the other classes get fresh symbols. P stays v1-periodic (line-periodic in
// quad-I) while acquiring defects w.r.t. the lattice in the corners.
inline Array2D<int> lattice_with_line_overlay(int m, Vec2 v1, Vec2 v2, int alpha, std::uint64_t seed) {
    Array2D<int> A = lattice(m, v1, v2, alpha, seed);
    std::mt19937_64 rng(seed ^ 0x1234567ULL);
    gp::View inner = top_inner(m);
    int g = std::gcd(gp::iabs(v1.r), gp::iabs(v1.c));
    Vec2 v0{v1.r / g, v1.c / g};
    auto fdiv = [](int a, int b) { int q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q; };
    auto cls = [&](Vec2 u) { long cr = gp::cross(u, v0); int t = v0.r != 0 ? fdiv(u.r, v0.r) : fdiv(u.c, v0.c); return cr * 64 + ((t % g) + g) % g; };
    std::set<long> touching;
    for (int r = inner.r0; r < inner.r0 + inner.size; ++r) for (int c = inner.c0; c < inner.c0 + inner.size; ++c) touching.insert(cls({r, c}));
    std::unordered_map<long, int> sym;
    for (int r = 0; r < m; ++r) for (int c = 0; c < m; ++c) {
        long k = cls({r, c});
        if (touching.count(k)) continue;
        auto it = sym.find(k);
        if (it == sym.end()) it = sym.emplace(k, alpha + 1 + (int)(rng() % alpha)).first;
        A.at(r, c) = it->second;
    }
    return A;
}

} // namespace gen

// ---------------------------------------------------------------- Milestone 3 generators
namespace gen {

// Line-periodic array (direction v, any quadrant) plus random point defects outside the top
// inner view: P^{t-1} stays line-periodic, P^t becomes nonperiodic or keeps part of the line.
inline Array2D<int> line_with_point_defects(int m, Vec2 v, int alpha, int count, std::uint64_t seed) {
    Array2D<int> A = line_dir(m, v, alpha, seed);
    std::mt19937_64 rng(seed ^ 0x77777ULL);
    gp::View inner = top_inner(m);
    int planted = 0, guard = 0;
    while (planted < count && guard++ < 100000) {
        Vec2 p{(int)(rng() % m), (int)(rng() % m)};
        if (inner.contains(p)) continue;
        A.at(p) = alpha + 1 + (int)(rng() % 3); ++planted;
    }
    return A;
}

// Lattice whose quad-II basis vector is valid for P^t but not for P^{t-1}: P^{t-1} is
// line-periodic in quad-I (v1), P^t is lattice-periodic -> Step B3 handoff. quadII_first swaps
// the roles (line-periodic in quad-II with a long quad-I vector).
inline Array2D<int> lattice_long_second(int m, std::mt19937_64& rng, bool quadII_first, Vec2& v1, Vec2& v2) {
    int K = gp::View{0, 0, m}.K(), Kp = top_inner(m).K();
    if (quadII_first) {
        v2 = Vec2{-(int)(rng() % Kp) - 1, (int)(rng() % (Kp + 1))};                 // short quad-II
        v1 = Vec2{Kp + 1 + (int)(rng() % (K - Kp)), Kp + 1 + (int)(rng() % (K - Kp))}; // long quad-I
    } else {
        v1 = Vec2{(int)(rng() % (Kp + 1)), (int)(rng() % Kp) + 1};                    // short quad-I
        v2 = Vec2{-(Kp + 1 + (int)(rng() % (K - Kp))), Kp + 1 + (int)(rng() % (K - Kp))}; // long quad-II
    }
    return lattice(m, v1, v2, 5, rng());
}

// Radiant-periodic P^{t-1}: lattice everywhere; defects at (and next to) two opposite corners
// of the top inner view I1 -- bottom-left and top-right (radiant in quad-I) or top-left and
// bottom-right (radiant in quad-II). Corner defects kill every quad-II (resp. quad-I) period of
// I1 but no quad-I (resp. quad-II) period with both coordinates nonzero (Fig. 7 of the paper).
// Optionally more random defects outside I1.
inline Array2D<int> radiant_stage(int m, Vec2 v1, Vec2 v2, int alpha, bool quadII_variant, int outer_defects, std::uint64_t seed) {
    Array2D<int> A = lattice(m, v1, v2, alpha, seed);
    std::mt19937_64 rng(seed ^ 0x5151ULL);
    gp::View I1 = top_inner(m);
    int r0 = I1.r0, c0 = I1.c0, r1 = I1.r0 + I1.size - 1, c1 = I1.c0 + I1.size - 1;
    std::vector<Vec2> corners = quadII_variant ? std::vector<Vec2>{{r0, c0}, {r1, c1}} : std::vector<Vec2>{{r1, c0}, {r0, c1}};
    int spread = (int)(rng() % 2);
    for (Vec2 p : corners) {
        A.at(p) = alpha + 1 + (int)(rng() % 3);
        if (spread) {   // a second defect adjacent along the edge
            Vec2 q = (p.r == r0) ? Vec2{r0, p.c == c0 ? c0 + 1 : c1 - 1} : Vec2{r1, p.c == c0 ? c0 + 1 : c1 - 1};
            A.at(q) = alpha + 1 + (int)(rng() % 3);
        }
    }
    int planted = 0, guard = 0;
    while (planted < outer_defects && guard++ < 100000) {
        Vec2 p{(int)(rng() % m), (int)(rng() % m)};
        if (I1.contains(p)) continue;
        A.at(p) = alpha + 1 + (int)(rng() % 3); ++planted;
    }
    return A;
}

} // namespace gen
