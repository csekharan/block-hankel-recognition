#pragma once
// Witness tables for C_I (r in [0,K], c in [1,K]) and C_II (r in [-K,-1], c in [0,K]).
// Witness points are stored in ABSOLUTE coordinates of P (design choice 1).
#include "vec2.hpp"
#include <vector>
#include <optional>
#include <cstdint>
#include <cassert>
#include <algorithm>

namespace gp {

enum class Status : std::uint8_t { Unknown, Period, Witness };

struct Witness {
    Status s = Status::Unknown;
    Vec2 w{};
};

struct WitnessTable {
    int cap = 0;                         // K of the largest stage this table serves
    std::vector<Witness> qI, qII;
    explicit WitnessTable(int cap_ = 0)
        : cap(imax(cap_, 0)),
          qI((std::size_t)(cap + 1) * cap),
          qII((std::size_t)cap * (cap + 1)) {}

    bool in_range(Vec2 v) const {
        if (quadI(v)) return v.r <= cap && v.c <= cap;
        if (quadII(v)) return -v.r <= cap && v.c <= cap;
        return false;
    }
    Witness& at(Vec2 v) {
        assert(in_range(v));
        if (quadI(v)) return qI[(std::size_t)v.r * cap + (v.c - 1)];
        return qII[(std::size_t)(-v.r - 1) * (cap + 1) + v.c];
    }
    const Witness& at(Vec2 v) const {
        assert(in_range(v));
        if (quadI(v)) return qI[(std::size_t)v.r * cap + (v.c - 1)];
        return qII[(std::size_t)(-v.r - 1) * (cap + 1) + v.c];
    }
    // Status of v or of its canonical representative -v.
    Status status(Vec2 v) const {
        Vec2 u = canon(v);
        if (!in_range(u)) return Status::Unknown;
        return at(u).s;
    }
    // A witness against v (any quadrant): x with P[x] != P[x - v].
    // If w is a witness against -v then w + v is a witness against v.
    std::optional<Vec2> witness_against(Vec2 v) const {
        Vec2 u = canon(v);
        if (!in_range(u)) return std::nullopt;
        const Witness& e = at(u);
        if (e.s != Status::Witness) return std::nullopt;
        return quadI_or_II(v) ? e.w : e.w + v;
    }
    // Record x as a witness against v (any quadrant).
    void set_witness(Vec2 v, Vec2 x) {
        if (quadI_or_II(v)) at(v) = {Status::Witness, x};
        else at(-v) = {Status::Witness, x - v};
    }
    void set_period(Vec2 v) { at(canon(v)) = {Status::Period, {}}; }

    // Enumerate C_I / C_II up to K in the length-lex orders.
    static std::vector<Vec2> boxI(int K) {
        std::vector<Vec2> out;
        for (int r = 0; r <= K; ++r) for (int c = 1; c <= K; ++c) out.push_back({r, c});
        std::sort(out.begin(), out.end(), LessI{});
        return out;
    }
    static std::vector<Vec2> boxII(int K) {
        std::vector<Vec2> out;
        for (int r = -K; r <= -1; ++r) for (int c = 0; c <= K; ++c) out.push_back({r, c});
        std::sort(out.begin(), out.end(), LessII{});
        return out;
    }
    // Sub-box Q_X^k (paper labels) of C_X^K relative to the previous box C_X^Kp.
    // C_I: Q1 = C_I^{t-1} (small r, small c), Q2 large r/small c, Q3 large/large, Q4 small r/large c.
    // C_II: Q2 = C_II^{t-1} (|r| small, c small), Q1 |r| large/c small, Q4 large/large, Q3 |r| small/c large.
    static std::vector<Vec2> subbox(bool quad_one, int k, int K, int Kp) {
        std::vector<Vec2> out;
        auto small_r = [&](int r) { return iabs(r) <= Kp; };
        auto small_c = [&](int c) { return c <= Kp; };
        if (quad_one) {
            for (int r = 0; r <= K; ++r) for (int c = 1; c <= K; ++c) {
                bool sr = small_r(r), sc = small_c(c);
                int q = (sr && sc) ? 1 : (!sr && sc) ? 2 : (!sr && !sc) ? 3 : 4;
                if (q == k) out.push_back({r, c});
            }
            std::sort(out.begin(), out.end(), LessI{});
        } else {
            for (int r = -K; r <= -1; ++r) for (int c = 0; c <= K; ++c) {
                bool sr = small_r(r), sc = small_c(c);
                int q = (sr && sc) ? 2 : (!sr && sc) ? 1 : (!sr && !sc) ? 4 : 3;
                if (q == k) out.push_back({r, c});
            }
            std::sort(out.begin(), out.end(), LessII{});
        }
        return out;
    }
};

} // namespace gp
