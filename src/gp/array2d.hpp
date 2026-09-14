#pragma once
// Array2D<T> (square, flat row-major), View (centered nested sub-squares), Frame (index transforms).
#include "vec2.hpp"
#include <vector>
#include <cassert>

namespace gp {

template <class T>
struct Array2D {
    int n = 0;
    std::vector<T> a;
    Array2D() = default;
    explicit Array2D(int n_, T init = T{}) : n(n_), a((std::size_t)n_ * n_, init) {}
    T& at(int r, int c) { return a[(std::size_t)r * n + c]; }
    const T& at(int r, int c) const { return a[(std::size_t)r * n + c]; }
    T& at(Vec2 p) { return at(p.r, p.c); }
    const T& at(Vec2 p) const { return at(p.r, p.c); }
    bool contains(Vec2 p) const { return p.r >= 0 && p.r < n && p.c >= 0 && p.c < n; }
    Array2D transposed() const {
        Array2D B(n);
        for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c) B.at(c, r) = at(r, c);
        return B;
    }
};

// A square sub-array of P in absolute coordinates.
struct View {
    int r0 = 0, c0 = 0, size = 0;
    constexpr bool operator==(const View&) const = default;
    bool contains(Vec2 p) const {
        return p.r >= r0 && p.r < r0 + size && p.c >= c0 && p.c < c0 + size;
    }
    Vec2 origin() const { return {r0, c0}; }
    Vec2 local(Vec2 p) const { return {p.r - r0, p.c - c0}; }
    Vec2 abs(Vec2 q) const { return {q.r + r0, q.c + c0}; }
    int half() const { return (size + 1) / 2; }          // ceil(size/2)
    int K() const { return ceil_div(size, 4) - 1; }        // candidates have |v| <= K
    // Paper quadrant labels: I upper-left, II lower-left, III lower-right, IV upper-right.
    bool in_quadrant(Vec2 p, int k) const {
        Vec2 q = local(p); int h = half();
        switch (k) {
            case 1: return q.r < h && q.c < h;
            case 2: return q.r >= size - h && q.c < h;
            case 3: return q.r >= size - h && q.c >= size - h;
            case 4: return q.r < h && q.c >= size - h;
        }
        return false;
    }
    // doubled Chebyshev distance to the center ((size-1)/2, (size-1)/2)
    int d2(Vec2 p) const {
        Vec2 q = local(p);
        return imax(iabs(2 * q.r - (size - 1)), iabs(2 * q.c - (size - 1)));
    }
    int dist_backdiag(Vec2 p) const { Vec2 q = local(p); return iabs(q.r + q.c - (size - 1)); }
    int dist_maindiag(Vec2 p) const { Vec2 q = local(p); return iabs(q.r - q.c); }
    // Centered nesting (Definition 2.1 of the mirrored-lemmas note):
    // s = least integer >= ceil(size/2) with size - s even; offset o = (size - s)/2.
    View inner() const {
        int s = (size + 1) / 2;
        if ((size - s) & 1) ++s;
        int o = (size - s) / 2;
        return {r0 + o, c0 + o, s};
    }
};

// Affine index transform p -> (a r + b c + tr, c r + d c + tc), restricted to
// rotations/transpositions of the m x m array (det = +-1).
struct Frame {
    int a = 1, b = 0, c = 0, d = 1, tr = 0, tc = 0;
    Vec2 map(Vec2 p) const { return {a * p.r + b * p.c + tr, c * p.r + d * p.c + tc}; }
    Vec2 mapv(Vec2 v) const { return {a * v.r + b * v.c, c * v.r + d * v.c}; }
    Frame inverse() const {
        int det = a * d - b * c;                // +-1
        int ia = d * det, ib = -b * det, ic = -c * det, id = a * det;
        return {ia, ib, ic, id, -(ia * tr + ib * tc), -(ic * tr + id * tc)};
    }
    View map_view(const View& v) const {
        Vec2 p1 = map(v.origin()), p2 = map({v.r0 + v.size - 1, v.c0 + v.size - 1});
        return {imin(p1.r, p2.r), imin(p1.c, p2.c), v.size};
    }
    // composition: (g.then_apply(*this)) p = g(this(p))  ->  g o this
    Frame then(const Frame& g) const {
        return {g.a * a + g.b * c, g.a * b + g.b * d, g.c * a + g.d * c, g.c * b + g.d * d,
                g.a * tr + g.b * tc + g.tr, g.c * tr + g.d * tc + g.tc};
    }
    static Frame identity() { return {}; }
    static Frame rot90(int m) { return {0, -1, 1, 0, m - 1, 0}; }     // (i,j) -> (m-1-j, i); L(r,c) = (-c, r)
    static Frame rot180(int m) { return {-1, 0, 0, -1, m - 1, m - 1}; }
    static Frame rot270(int m) { return {0, 1, -1, 0, 0, m - 1}; }
    static Frame transpose() { return {0, 1, 1, 0, 0, 0}; }
};

} // namespace gp
