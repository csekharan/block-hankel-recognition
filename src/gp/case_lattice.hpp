#pragma once
// Section 4.2: P^{t-1} is lattice-periodic with basis (vI, vII). Steps A1-A5.
// Every witness written here is verified by one symbol comparison before it is stored, so a
// wrong reading of a lemma can only cost a fallback (counted), never a wrong table entry.
#include "array2d.hpp"
#include "witness.hpp"
#include "oracle.hpp"
#include "line.hpp"
#include "lattice.hpp"
#include "instrument.hpp"
#include <numeric>
#include <tuple>
#include <optional>

namespace gp {

// store x as witness against v if P[x] != P[x - v] with both points in cur
template <class T>
bool try_witness(const Array2D<T>& A, const View& cur, WitnessTable& tab, Vec2 v, Vec2 x) {
    if (!cur.contains(x) || !cur.contains(x - v)) return false;
    if (A.at(x) == A.at(x - v)) return false;
    tab.set_witness(v, x);
    return true;
}

// Frame-coordinate access: the case code reasons in a rotated/transposed picture of cur
// (a symmetry of the centered squares), all storage stays in absolute coordinates.
template <class T>
struct FrameCtx {
    const Array2D<T>& A; const View& cur; const View& prev;
    const DefectGrid& D; const Lattice& L; WitnessTable& tab;
    Frame Fi;                                    // frame -> absolute
    const T& at(Vec2 pf) const { return A.at(Fi.map(pf)); }
    bool defect(Vec2 pf) const { return D.is_defect(Fi.map(pf)); }
    bool lattice(Vec2 vf) const { return L.is_lattice_point(Fi.mapv(vf)); }
    Status status(Vec2 vf) const { return tab.status(Fi.mapv(vf)); }
    void set_period(Vec2 vf) { tab.set_period(Fi.mapv(vf)); }
    bool try_witness(Vec2 vf, Vec2 xf) { return gp::try_witness(A, cur, tab, Fi.mapv(vf), Fi.map(xf)); }
    FrameCtx composed(const Frame& g) const {    // frame' with g: frame' -> frame
        FrameCtx c = *this; c.Fi = g.then(Fi); return c;
    }
};

// Vectors of the frame-quad-I box that are compatible with the quadrant-II defects (frame
// quadrant II = lower-left of cur). Candidates must be strict quad-I lattice points with
// status Unknown. Incompatible candidates receive witnesses. If last_per_row is set, only the
// rightmost defect of every quadrant-II row is used (the set L of Step A5.2).
template <class T>
std::vector<Vec2> compat_quadII(FrameCtx<T>& ctx, const std::vector<Vec2>& cands, bool last_per_row) {
    const View& cur = ctx.cur;
    int mh = cur.size, h = cur.half();
    std::vector<int> d(mh, -1);
    std::vector<std::vector<int>> cols(last_per_row ? 0 : mh);
    for (int i = mh - h; i < mh; ++i)
        for (int j = 0; j < h; ++j)
            if (ctx.defect(cur.abs({i, j}))) { d[i] = j; if (!last_per_row) cols[i].push_back(j); }
    int h0 = -1, hp = -1, row_hp = -1;
    for (int i = 0; i < mh; ++i) if (d[i] >= 0) { if (h0 < 0) h0 = i; if (d[i] > hp) { hp = d[i]; row_hp = i; } }
    if (h0 < 0) return cands;                          // no quadrant-II defects
    auto P = [&](Vec2 loc) -> const T& { return ctx.at(cur.abs(loc)); };
    auto u = [&](int i) { return Vec2{i, d[i]}; };     // local
    // Early exit of Step A5.2 (also valid in A5.1): if the first defect row lies below P^{t-1}
    // and its rightmost defect is inside P^{t-1}'s column band, u_h - v is in quadrant II above
    // every defect row, so u_h is a witness against every candidate. Without this the case-2.1
    // claim below is only valid when d_h < o (the paper's "otherwise").
    int o = ctx.prev.r0 - cur.r0, s = ctx.prev.size;
    bool early = (h0 >= o + s) && (d[h0] >= o);
    std::vector<Vec2> out;
    for (Vec2 v : cands) {
        if (ctx.status(v) != Status::Unknown) continue;
        int r = v.r, c = v.c;
        if (early) {
            if (ctx.try_witness(v, cur.abs(u(h0)))) continue;
            gp_soft_check(false, "A5:early-exit");
        }
        if (r >= mh - h0 && c > hp) { out.push_back(v); continue; }          // Lemma 10: unaffected
        // EASY-WITNESS (verified)
        bool done = false, claimed = false;
        if (r >= mh - h0) {                                                    // case 1: c <= hp
            claimed = true;
            done = ctx.try_witness(v, cur.abs(Vec2{row_hp, hp}));
        } else {
            int dh = d[h0], dr = d[h0 + r];
            if (c != dr - dh) {
                claimed = true;
                if (c > dr - dh) done = ctx.try_witness(v, cur.abs(u(h0) + v));
                else done = ctx.try_witness(v, cur.abs(u(h0 + r)));
            }
        }
        if (claimed && !done) gp_soft_check(false, "A5:easy-witness");
        if (done) continue;
        // direct compatibility check against the chosen defect set
        bool bad = false;
        for (int i = h0; i < mh && !bad; ++i) {
            if (d[i] < 0) continue;
            auto test = [&](Vec2 loc) {
                Vec2 p = loc + v, q = loc - v;
                if (cur.contains(cur.abs(p)) && !(P(loc) == P(p))) { bad = ctx.try_witness(v, cur.abs(p)); return; }
                if (cur.contains(cur.abs(q)) && !(P(loc) == P(q))) { bad = ctx.try_witness(v, cur.abs(loc)); return; }
            };
            if (last_per_row) test(u(i));
            else for (int j : cols[i]) { test({i, j}); if (bad) break; }
        }
        if (!bad) out.push_back(v);
    }
    return out;
}

// Direct compatibility of v (frame vector) with every defect of cur; returns true if v is
// a period (and marks it), false if a witness was stored.
template <class T>
bool settle_direct(FrameCtx<T>& ctx, Vec2 vf) {
    Vec2 v = ctx.Fi.mapv(vf);
    for (Vec2 uabs : ctx.D.list) {
        if (ctx.cur.contains(uabs + v) && !(ctx.A.at(uabs) == ctx.A.at(uabs + v))) { ctx.tab.set_witness(v, uabs + v); return false; }
        if (ctx.cur.contains(uabs - v) && !(ctx.A.at(uabs) == ctx.A.at(uabs - v))) { ctx.tab.set_witness(v, uabs); return false; }
    }
    ctx.tab.set_period(v);
    return true;
}

// Candidates of side X (absolute), expressed as strict frame-quad-I vectors.
inline std::vector<Vec2> frame_candidates(bool sideI, int K, const Frame& F, const Lattice& L, const WitnessTable& tab) {
    std::vector<Vec2> out;
    for (Vec2 v : (sideI ? WitnessTable::boxI(K) : WitnessTable::boxII(K))) {
        if (tab.at(v).s != Status::Unknown) continue;
        if (v.r == 0 || v.c == 0) continue;            // axis vectors were decided by LINE in A3
        if (!L.is_lattice_point(v)) continue;
        Vec2 vf = F.mapv(v);
        if (!quadI(vf)) vf = -vf;
        out.push_back(vf);
    }
    std::sort(out.begin(), out.end(), LessI{});
    return out;
}

// Step A5 for the side whose vectors are frame-quad-I; frame chosen so that the nearest
// defect lies in frame quadrant II or IV.
template <class T>
void step_A5(FrameCtx<T>& ctx, bool sideI, int K) {
    const View& cur = ctx.cur;
    bool inq[5] = {false, false, false, false, false};
    for (Vec2 uabs : ctx.D.list) {
        Vec2 uf = ctx.Fi.inverse().map(uabs);
        for (int k = 1; k <= 4; ++k) if (cur.in_quadrant(uf, k)) inq[k] = true;
    }
    bool adjacent = (inq[1] && inq[2]) || (inq[2] && inq[3]) || (inq[3] && inq[4]) || (inq[4] && inq[1]);
    Frame F = ctx.Fi.inverse();
    std::vector<Vec2> cands = frame_candidates(sideI, K, F, ctx.L, ctx.tab);
    Frame Tr = Frame::transpose();
    FrameCtx<T> ctxT = ctx.composed(Tr);
    auto transposed = [&](const std::vector<Vec2>& vs) {
        std::vector<Vec2> o; for (Vec2 v : vs) if (ctx.status(v) == Status::Unknown) o.push_back(Tr.mapv(v));
        std::sort(o.begin(), o.end(), LessI{}); return o;
    };
    if (!adjacent) {
        gp_path("A5.1");
        std::vector<Vec2> VII = compat_quadII(ctx, cands, false);
#ifdef GP_DEBUG_A5
        std::fprintf(stderr, "A5.1: cands=%zu VII=%zu sideI=%d Fi=(%d,%d,%d,%d,%d,%d)\n", cands.size(), VII.size(), (int)sideI, ctx.Fi.a, ctx.Fi.b, ctx.Fi.c, ctx.Fi.d, ctx.Fi.tr, ctx.Fi.tc);
        for (Vec2 v : cands) std::fprintf(stderr, "   cand (%d,%d) status=%d\n", v.r, v.c, (int)ctx.status(v));
#endif
        std::vector<Vec2> VIV = compat_quadII(ctxT, transposed(VII), false);
        for (Vec2 vT : VIV) ctxT.set_period(vT);       // compatible with II and IV, no other defects
    } else {
        std::vector<Vec2> VL = compat_quadII(ctx, cands, true);
        std::vector<Vec2> ULt = compat_quadII(ctxT, transposed(VL), true);
        std::vector<Vec2> U; for (Vec2 vT : ULt) U.push_back(Tr.mapv(vT));
        if (U.empty()) { gp_path("A5.2:empty"); return; }
        bool line = true;
        for (Vec2 v : U) if (!collinear(v, U[0])) { line = false; break; }
        if (line) {
            gp_path("A5.2:line");
            Vec2 dir = ctx.Fi.mapv(U[0]);
            int g = std::gcd(iabs(dir.r), iabs(dir.c));
            dir = {dir.r / g, dir.c / g};
            line_procedure(ctx.A, cur, dir, ctx.tab, K);
            for (Vec2 v : U) if (ctx.status(v) == Status::Unknown) { gp_soft_check(false, "A5.2:line-left-unknown"); settle_direct(ctx, v); }
        } else {
            gp_path("A5.2:independent");
            std::optional<Vec2> uc; int best = 1 << 30;
            for (Vec2 uabs : ctx.D.list) {
                Vec2 uf = F.map(uabs);
                if (!(cur.in_quadrant(uf, 1) || cur.in_quadrant(uf, 3))) continue;
                int dd = cur.d2(uf);
                if (dd < best || (dd == best && std::tuple(uf.r, uf.c) < std::tuple(uc->r, uc->c))) { best = dd; uc = uf; }
            }
            // u'_c in quadrant III: u'_c - v (up-left, toward the interior) is the non-defect and
            // u'_c the witness point; u'_c in quadrant I: the mirror image, u'_c + v is the
            // non-defect and the witness point (the paper writes only the quadrant-III case).
            bool ucIII = uc && cur.in_quadrant(*uc, 3);
            for (Vec2 v : U) {
                if (ctx.status(v) != Status::Unknown) continue;
                if (uc && ctx.try_witness(v, ucIII ? *uc : *uc + v)) continue;
                if (uc && ctx.try_witness(v, ucIII ? *uc + v : *uc)) { gp_soft_check(false, "A5.2:independent-direction"); continue; }
                gp_soft_check(false, "A5.2:independent-witness");
                settle_direct(ctx, v);
            }
        }
    }
}

// ---- the case ---------------------------------------------------------------------------
// from_B3: entered from Step B3 of the line-periodic case with basis (vI, vhat_II); A1 is
// skipped because every nonlattice vector already has a witness, and lattice points that lost a
// duel in B3.1 keep their witnesses.
template <class T>
bool case_lattice(const Array2D<T>& A, const View& cur, const View& prev, WitnessTable& tab,
                  int K, int Kp, Vec2 vI, Vec2 vII, bool from_B3 = false) {
    gp_path(from_B3 ? "4.2:from-B3" : "4.2");
    Lattice L(vI, vII);
    int s = prev.size, o = prev.r0 - cur.r0;
    // step 0: undecided lattice points are Unknown; nonlattice points must have witnesses
    for (bool q1 : {true, false})
        for (Vec2 v : (q1 ? WitnessTable::boxI(K) : WitnessTable::boxII(K))) {
            Status st = tab.at(v).s;
            if (L.is_lattice_point(v)) { if (st != Status::Witness) tab.at(v) = {Status::Unknown, {}}; }
            else if ((from_B3 || len(v) <= Kp) && st != Status::Witness) {
                gp_soft_check(false, from_B3 ? "B3:nonlattice-witness" : "Cor3:nonlattice-has-witness");
                tab.at(v) = brute_force_witness(A, cur, v);
            }
        }

    // ---- A1: nonlattice points of the new sub-boxes ----------------------------------------
    if (!from_B3) {
    gp_path("A1");
        std::vector<Vec2> prevbox = WitnessTable::boxI(Kp);
        for (Vec2 v : WitnessTable::boxII(Kp)) prevbox.push_back(v);
        RepMap repC = RepMap::build(L, prevbox.begin(), prevbox.end());
        int hp = prev.half();
        View qIII{prev.r0 + s - hp, prev.c0 + s - hp, hp}, qIV{prev.r0, prev.c0 + s - hp, hp};
        gp_soft_check(L.covered_by(2 * Kp + 1, Kp) && L.covered_by(hp, hp), "A1:covering");
        RepMap repIII = RepMap::build(L, qIII), repIV = RepMap::build(L, qIV);
        for (bool q1 : {true, false})
            for (Vec2 v : (q1 ? WitnessTable::boxI(K) : WitnessTable::boxII(K))) {
                if (len(v) <= Kp || L.is_lattice_point(v)) continue;
                bool ok = false;
                auto u = repC.find(L, v);
                if (u && tab.at(*u).s == Status::Witness) {
                    Vec2 w = tab.at(*u).w;
                    auto wp = (q1 ? repIII : repIV).find(L, w);
                    if (wp) ok = try_witness(A, cur, tab, v, *wp);
                }
                if (!ok) { gp_soft_check(false, "A1:witness"); tab.at(v) = brute_force_witness(A, cur, v); }
            }
    }
    // ---- A2: defects -----------------------------------------------------------------------
    gp_soft_check(L.covered_by(s, s), "A2:covering");
    DefectGrid D = find_defects(A, cur, prev, L);
    if (D.list.empty()) {
        gp_path("A2:no-defects");
        for (bool q1 : {true, false})
            for (Vec2 v : (q1 ? WitnessTable::boxI(K) : WitnessTable::boxII(K)))
                if (tab.at(v).s == Status::Unknown) tab.set_period(v);
        return true;
    }
    // ---- A3: axis periods ------------------------------------------------------------------
    line_procedure(A, cur, Vec2{1, 0}, tab, K);
    line_procedure(A, cur, Vec2{0, 1}, tab, K);
    bool has_vert = false, has_horiz = false;
    for (int k = 1; k <= K; ++k) { if (tab.status({-k, 0}) == Status::Period) has_vert = true; if (tab.status({0, k}) == Status::Period) has_horiz = true; }
    gp_soft_check(!(has_vert && has_horiz), "Lemma13:both-axes");
    auto settle_all_undecided = [&](auto&& choose) {     // choose(v) -> witness point candidates
        for (bool q1 : {true, false})
            for (Vec2 v : (q1 ? WitnessTable::boxI(K) : WitnessTable::boxII(K))) {
                if (tab.at(v).s != Status::Unknown) continue;
                auto [first, second] = choose(v);
                if (try_witness(A, cur, tab, v, first) || try_witness(A, cur, tab, v, second)) continue;
                gp_soft_check(false, "A3/A4:witness");
                tab.at(v) = brute_force_witness(A, cur, v);
            }
    };
    if (has_vert || has_horiz) {
        gp_path("A3");
        bool vert = has_vert;
        // u_c: defect with row (vertical case) / column (horizontal case) inside P^{t-1}'s band,
        // nearest to the band in the other coordinate (repaired Step A3, item 2).
        std::optional<Vec2> uc; int best = 1 << 30;
        for (Vec2 u : D.list) {
            int along = vert ? u.r : u.c, across = vert ? u.c : u.r;
            int lo = vert ? prev.r0 : prev.c0;
            if (along < lo || along >= lo + s) continue;
            int dist = iabs(2 * (across - (vert ? cur.c0 : cur.r0)) - (cur.size - 1));
            if (dist < best) { best = dist; uc = u; }
        }
        if (!gp_soft_check(uc.has_value(), "A3:uc-in-band")) uc = D.list.front();
        int band_lo = vert ? prev.c0 : prev.r0;
        bool before = (vert ? uc->c : uc->r) < band_lo;
        settle_all_undecided([&](Vec2 v) {
            // move toward the band: columns always increase with +v (c >= 1 for non-axis);
            // rows increase with +v for quad-I and with -v for quad-II.
            // With +v the non-defect is uc+v and the witness point is uc+v; with -v the
            // non-defect is uc-v and the witness point is uc. Second entry: the other option.
            bool plus = vert ? before : (quadI(v) ? before : !before);
            return std::pair<Vec2, Vec2>{plus ? *uc + v : *uc, plus ? *uc : *uc + v};
        });
        return true;
    }
    // ---- A4: nearest defect settles one side ------------------------------------------------
    gp_path("A4");
    Vec2 uc = D.list.front(); int best = cur.d2(uc);
    for (Vec2 u : D.list) { int dd = cur.d2(u); if (dd < best || (dd == best && std::tuple(u.r, u.c) < std::tuple(uc.r, uc.c))) { best = dd; uc = u; } }
    int q = 1; while (q < 4 && !cur.in_quadrant(uc, q)) ++q;
    bool free_sideII = (q == 2 || q == 4);
    int sign = (q == 1 || q == 2) ? 1 : -1;
    for (Vec2 v : (free_sideII ? WitnessTable::boxII(K) : WitnessTable::boxI(K))) {
        if (tab.at(v).s != Status::Unknown) continue;
        Vec2 x = uc + sign * v;
        Vec2 wp = sign > 0 ? x : uc;
        if (try_witness(A, cur, tab, v, wp)) continue;
        Vec2 wp2 = sign > 0 ? uc : uc + v;                    // the other direction
        if (try_witness(A, cur, tab, v, wp2)) { gp_soft_check(false, "A4:direction"); continue; }
        gp_soft_check(false, "A4:witness");
        tab.at(v) = brute_force_witness(A, cur, v);
    }
    // ---- A5: the other side, in a frame where u_c is in quadrant II or IV -------------------
    Frame F = free_sideII ? Frame::identity() : Frame::rot90(A.n);   // rot90 about the center of P
    FrameCtx<T> ctx{A, cur, prev, D, L, tab, F.inverse()};
    step_A5(ctx, free_sideII, K);   // compute the side A4 did not settle
    (void)o;
    return true;
}

} // namespace gp
