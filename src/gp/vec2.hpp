#pragma once
// Points and vectors in the Galil-Park convention: (row, col), row 0 at top.
#include <tuple>
#include <cstdint>
#include <cstddef>
#include <functional>

namespace gp {

constexpr int iabs(int x) { return x < 0 ? -x : x; }
constexpr int imax(int a, int b) { return a < b ? b : a; }
constexpr int imin(int a, int b) { return a < b ? a : b; }
constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct Vec2 {
    int r = 0, c = 0;
    constexpr bool operator==(const Vec2&) const = default;
};
constexpr Vec2 operator+(Vec2 a, Vec2 b) { return {a.r + b.r, a.c + b.c}; }
constexpr Vec2 operator-(Vec2 a, Vec2 b) { return {a.r - b.r, a.c - b.c}; }
constexpr Vec2 operator-(Vec2 a) { return {-a.r, -a.c}; }
constexpr Vec2 operator*(int k, Vec2 a) { return {k * a.r, k * a.c}; }

constexpr int len(Vec2 v) { return imax(iabs(v.r), iabs(v.c)); }

// Quadrant classes of vectors (paper, Section 2). The zero vector is in none.
constexpr bool quadI(Vec2 v)   { return v.r >= 0 && v.c > 0; }
constexpr bool quadII(Vec2 v)  { return v.r < 0 && v.c >= 0; }
constexpr bool quadIII(Vec2 v) { return v.r <= 0 && v.c < 0; }
constexpr bool quadIV(Vec2 v)  { return v.r > 0 && v.c <= 0; }
constexpr bool quadI_or_II(Vec2 v) { return quadI(v) || quadII(v); }
// canonical representative: v itself if quad-I/II, otherwise -v
constexpr Vec2 canon(Vec2 v) { return quadI_or_II(v) ? v : -v; }

// partial orders: u <_I v iff v-u is quad-I; u <_II v iff v-u is quad-II
constexpr bool precI(Vec2 u, Vec2 v)  { return quadI(v - u); }
constexpr bool precII(Vec2 u, Vec2 v) { return quadII(v - u); }

constexpr long cross(Vec2 a, Vec2 b) { return (long)a.r * b.c - (long)a.c * b.r; }
constexpr bool collinear(Vec2 a, Vec2 b) { return cross(a, b) == 0; }

// Length-lexicographic total orders (Finding 2: quad-II uses |r|).
struct LessI {
    constexpr bool operator()(Vec2 a, Vec2 b) const {
        return std::tuple(len(a), a.r, a.c) < std::tuple(len(b), b.r, b.c);
    }
};
struct LessII {
    constexpr bool operator()(Vec2 a, Vec2 b) const {
        return std::tuple(len(a), a.c, iabs(a.r)) < std::tuple(len(b), b.c, iabs(b.r));
    }
};

struct Vec2Hash {
    std::size_t operator()(Vec2 v) const noexcept {
        return std::hash<long long>()(((long long)v.r << 32) ^ (unsigned)v.c);
    }
};

} // namespace gp
