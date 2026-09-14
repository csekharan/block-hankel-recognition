#pragma once
// Sections 4.3 (line-periodic P^{t-1}) and 4.4 (radiant-periodic P^{t-1}).
// The quad-I case is implemented; the quad-II case runs the same code in the rot90 frame
// (transfer principle). Candidate bookkeeping is in frame vectors; every table, array, duel and
// LINE operation is performed in absolute coordinates on the images of the frame vectors.
#include "array2d.hpp"
#include "witness.hpp"
#include "oracle.hpp"
#include "duel.hpp"
#include "line.hpp"
#include "lattice.hpp"
#include "case_lattice.hpp"
#include "classify.hpp"
#include "instrument.hpp"
#include <numeric>
#include <optional>

namespace gp {

template <class T>
struct LineCtx {
    const Array2D<T>& A; const View& cur; const View& prev; WitnessTable& tab;
    int K, Kp;
    Frame F, Fi;             // absolute -> frame, frame -> absolute
    Rule rule2;              // Rule 2 (identity frame) or Rule 2' (rot90 frame)
    Vec2 abs(Vec2 vf) const { return Fi.mapv(vf); }
    Status status(Vec2 vf) const { return tab.status(abs(vf)); }
    bool has_wit(Vec2 df) const { return tab.witness_against(abs(df)).has_value(); }
    void set_witness(Vec2 vf, Vec2 x) { tab.set_witness(abs(vf), x); }
    bool try_witness(Vec2 vf, Vec2 x) { return gp::try_witness(A, cur, tab, abs(vf), x); }
    void brute(Vec2 vf) {
        Witness w = brute_force_witness(A, cur, abs(vf));
        if (w.s == Status::Witness) tab.set_witness(abs(vf), w.w); else tab.set_period(abs(vf));
    }
    Witness brute_prev(Vec2 vf) const { return brute_force_witness(A, prev, abs(vf)); }
    // duel two frame candidates; false if there is no seed witness or no legal apex
    bool duel(Vec2 xf, Vec2 yf, Rule rule, bool& x_lost, bool& y_lost) {
        PairOutcome o = duel_pair(A, cur, tab, abs(xf), abs(yf), rule, &prev);
        if (!o.ok) return false;
        x_lost = o.wx.has_value(); y_lost = o.wy.has_value();
        if (o.wx) set_witness(xf, *o.wx);
        if (o.wy) set_witness(yf, *o.wy);
        return true;
    }
};

// Candidate reduction by duels (4.1 all-differences, B1.1/D1.1/B3.1 lines, radiant chains,
// D1.2 merge). A newcomer duels every survivor whose difference has a witness; survivors are
// returned in candidate order. Correctness never depends on the structure of the survivors.
template <class T>
std::vector<Vec2> reduce_generic(LineCtx<T>& c, const std::vector<Vec2>& cands, Rule rule, const char* tag) {
    std::vector<Vec2> S;
    for (Vec2 v : cands) {
        if (c.status(v) != Status::Unknown) continue;
        bool v_lost = false;
        for (std::size_t i = S.size(); i-- > 0 && !v_lost;) {
            Vec2 u = S[i];
            if (!c.has_wit(v - u)) continue;
            bool ul = false, vl = false;
            if (!c.duel(u, v, rule, ul, vl)) { gp_soft_check(false, tag); c.brute(v); v_lost = true; break; }
            if (ul) S.erase(S.begin() + (long)i);
            if (vl) v_lost = true;
        }
        if (!v_lost) S.push_back(v);
    }
    return S;
}

// kill every vector of a line S with the witness w of P^{t-1} against one of its vectors v0
// (Lemma 15 for quad-II lines, the B4 argument for quad-I lines); each claim is verified.
template <class T>
void kill_line(LineCtx<T>& c, const std::vector<Vec2>& S, Vec2 v0, Vec2 w, const char* tag) {
    for (Vec2 vp : S) {
        if (c.status(vp) != Status::Unknown) continue;
        if (c.try_witness(vp, w)) continue;
        if (c.try_witness(vp, w + c.abs(vp) - c.abs(v0))) continue;
        gp_soft_check(false, tag);
        c.brute(vp);
    }
}

struct B1Result {
    std::vector<std::vector<Vec2>> lines;   // surviving quad-II candidates per sub-box (periods of P^{t-1})
    std::optional<Vec2> vhat;               // shortest survivor
};

// Step B1: quad-II candidates of the three new sub-boxes. In the radiant case (Lemma 9) every
// sub-box is killed.
template <class T>
B1Result step_B1(LineCtx<T>& c, bool radiant) {
    B1Result R;
    for (int k : {1, 3, 4}) {
        std::vector<Vec2> S = reduce_generic(c, WitnessTable::subbox(false, k, c.K, c.Kp), Rule::Rule1, "B1.1:duel");
        if (S.empty()) continue;
        Vec2 v = S[0];                                   // shortest: candidates came in LessII order
        Witness w = c.brute_prev(v);
        if (w.s == Status::Witness) { gp_path("B1.2:kill"); kill_line(c, S, v, w.w, "B1.2:lemma15"); }
        else if (radiant) { gp_soft_check(false, "4.4:Lemma9"); for (Vec2 vp : S) c.brute(vp); }
        else { gp_path("B1.2:survive"); R.lines.push_back(S); }
    }
    for (auto& S : R.lines) for (Vec2 v : S)
        if (c.status(v) == Status::Unknown && (!R.vhat || LessII{}(v, *R.vhat))) R.vhat = v;
    return R;
}

// Step D1: D1.1 within the new quad-I sub-boxes (unless already done by B3.1), then D1.2 merges
// everything undecided in C_I^t; the survivors form a monotone line M.
template <class T>
std::vector<Vec2> step_D1(LineCtx<T>& c, bool do_D11) {
    if (do_D11)
        for (int k : {2, 3, 4}) reduce_generic(c, WitnessTable::subbox(true, k, c.K, c.Kp), Rule::Rule1, "D1.1:duel");
    std::vector<Vec2> merged;
    for (Vec2 v : WitnessTable::boxI(c.K)) if (c.status(v) == Status::Unknown) merged.push_back(v);   // LessI order
    std::vector<Vec2> M = reduce_generic(c, merged, Rule::Rule1, "D1.2:duel");
    for (std::size_t i = 0; i < M.size(); ++i)
        for (std::size_t j = i + 1; j < M.size(); ++j)
            if (!precI(M[i], M[j])) { gp_soft_check(false, "D1.2:monotone"); break; }
    return M;
}

// Step D2 with Rule 2 / Rule 2', then LINE along the surviving direction.
template <class T>
void step_D2(LineCtx<T>& c, const std::vector<Vec2>& M) {
    gp_path("D2");
    std::vector<Vec2> Vh;
    for (Vec2 vq : M) {
        if (c.status(vq) != Status::Unknown) continue;
        if (Vh.empty() || collinear(vq, Vh[0])) { Vh.push_back(vq); continue; }
        bool lost = false;
        for (std::size_t i = Vh.size(); i-- > 0 && !lost;) {
            Vec2 u = Vh[i];
            if (!c.has_wit(vq - u)) { gp_soft_check(false, "D2:seed-missing"); c.brute(vq); lost = true; break; }
            bool ul = false, vl = false;
            if (!c.duel(u, vq, c.rule2, ul, vl)) { gp_soft_check(false, "D2:duel"); c.brute(vq); lost = true; break; }
            if (ul) Vh.erase(Vh.begin() + (long)i);
            if (vl) lost = true;
        }
        if (!lost) Vh.push_back(vq);
    }
    if (Vh.empty()) return;
    Vec2 d = Vh[0];
    int g = std::gcd(iabs(d.r), iabs(d.c));
    d = {d.r / g, d.c / g};
    line_procedure(c.A, c.cur, c.abs(d), c.tab, c.K);
    for (Vec2 v : Vh) if (c.status(v) == Status::Unknown) { gp_soft_check(false, "D2:line-left-unknown"); c.brute(v); }
}

// Steps B3 / B4: survivors exist in C_II^t; vhat is the shortest quad-II period of P^{t-1}.
template <class T>
void step_B3_B4(LineCtx<T>& c, const B1Result& R, Vec2 vI_f) {
    Vec2 a = canon(c.abs(vI_f)), b = canon(c.abs(*R.vhat));
    Vec2 bI = quadI(a) ? a : b, bII = quadI(a) ? b : a;
    if (!gp_soft_check(quadI(bI) && quadII(bII), "B3:basis")) {
        for (bool q1 : {true, false}) for (Vec2 v : (q1 ? WitnessTable::boxI(c.K) : WitnessTable::boxII(c.K)))
            if (c.tab.at(v).s == Status::Unknown) c.tab.at(v) = brute_force_witness(c.A, c.cur, v);
        return;
    }
    Lattice L(bI, bII);
    // Lemma 16(2): surviving quad-II candidates are lattice points on (vI, vhat)
    for (auto& S : R.lines) for (Vec2 v : S)
        if (c.status(v) == Status::Unknown && !L.is_lattice_point(c.abs(v))) { gp_soft_check(false, "Lemma16:lattice"); c.brute(v); }
    // B3.1 (= D1.1)
    std::vector<std::vector<Vec2>> lines(5);
    for (int k : {2, 3, 4}) lines[k] = reduce_generic(c, WitnessTable::subbox(true, k, c.K, c.Kp), Rule::Rule1, "B3.1:duel");
    // B3.2: one point per line decides lattice membership of the whole line
    bool all_lattice = true;
    for (int k : {2, 3, 4}) {
        auto& S = lines[k];
        if (S.empty()) continue;
        bool lat = L.is_lattice_point(c.abs(S[0]));
        for (Vec2 v : S) if (L.is_lattice_point(c.abs(v)) != lat) { gp_soft_check(false, "B3.2:line-congruent"); lat = false; }
        if (!lat) all_lattice = false;
    }
    if (!all_lattice) {
        gp_path("B4");
        for (int k : {2, 4}) {                              // nonlattice lines in Q_I^2, Q_I^4: Theorem 1
            auto& S = lines[k];
            if (S.empty() || L.is_lattice_point(c.abs(S[0]))) continue;
            Vec2 v = S.back();                              // longest
            Witness w = c.brute_prev(v);
            if (w.s == Status::Witness) kill_line(c, S, v, w.w, "B4:Q2Q4-propagate");
            else { gp_soft_check(false, "B4:Theorem1"); for (Vec2 vp : S) c.brute(vp); }
            S.clear();
        }
        auto& S3 = lines[3];
        if (!S3.empty() && !L.is_lattice_point(c.abs(S3[0]))) {
            Vec2 v = S3.back();
            Witness w = c.brute_prev(v);
            if (w.s == Status::Witness) { kill_line(c, S3, v, w.w, "B4:Q3-propagate"); S3.clear(); }
            else {
                gp_path("B4:nonlattice-period");            // Lemma 24 special case
                bool only = true;
                for (Vec2 u : WitnessTable::boxII(c.K)) if (!(u == *R.vhat) && c.status(u) == Status::Unknown) only = false;
                bool inQ4 = false;
                for (Vec2 u : WitnessTable::subbox(false, 4, c.K, c.Kp)) if (u == *R.vhat) inQ4 = true;
                bool nopair = true;
                std::vector<Vec2> cI; for (Vec2 u : WitnessTable::boxI(c.K)) if (c.status(u) == Status::Unknown) cI.push_back(u);
                for (Vec2 u1 : cI) for (Vec2 u2 : cI) if (u1 - u2 == *R.vhat) nopair = false;
                gp_soft_check(only && inQ4 && nopair, "Lemma24");
                std::vector<Vec2> M = step_D1(c, false);
                step_D2(c, M);
                for (Vec2 u : WitnessTable::boxII(c.K)) if (c.status(u) == Status::Unknown) c.brute(u);   // vhat by symbol comparisons
                return;
            }
        }
    }
    gp_path("B3:lattice-handoff");
    case_lattice(c.A, c.cur, c.prev, c.tab, c.K, c.Kp, bI, bII, /*from_B3=*/true);
}

// ---- the case ---------------------------------------------------------------------------
template <class T>
bool case_line(const Array2D<T>& A, const View& cur, const View& prev, WitnessTable& tab,
               int K, int Kp, const Classification& cls) {
    bool quadII_case = (cls.cat == Category::LineII || cls.cat == Category::RadiantII);
    bool radiant = (cls.cat == Category::RadiantI || cls.cat == Category::RadiantII);
    gp_path(radiant ? "4.4" : "4.3");
    Frame F = quadII_case ? Frame::rot90(A.n) : Frame::identity();
    LineCtx<T> c{A, cur, prev, tab, K, Kp, F, F.inverse(), quadII_case ? Rule::Rule2II : Rule::Rule2I};
    // inherited periods of P^{t-1} are candidates again; witnesses stay
    for (bool q1 : {true, false})
        for (Vec2 v : (q1 ? WitnessTable::boxI(K) : WitnessTable::boxII(K)))
            if (tab.at(v).s != Status::Witness) tab.at(v) = {Status::Unknown, {}};
    Vec2 vI_f = quadII_case ? -F.mapv(*cls.vII) : *cls.vI;      // frame quad-I shortest period of P^{t-1}
    gp_soft_check(quadI(vI_f), "4.3:frame-vI");

    B1Result R = step_B1(c, radiant);
    if (radiant) {
        if (R.vhat) { gp_soft_check(false, "4.4:survivors"); for (auto& S : R.lines) for (Vec2 v : S) if (c.status(v) == Status::Unknown) c.brute(v); }
        gp_path("4.4:B2");
        std::vector<Vec2> M = step_D1(c, true);
        step_D2(c, M);
        return true;
    }
    if (!R.vhat) {
        gp_path("B2");
        std::vector<Vec2> M = step_D1(c, true);
        step_D2(c, M);
        return true;
    }
    gp_path("B3");
    step_B3_B4(c, R, vI_f);
    return true;
}

} // namespace gp
