#pragma once
// Stage driver: nested centered sub-squares, classification of P^{t-1}, case dispatch.
// Milestone 1 implements the base stage and Section 4.1 (P^{t-1} nonperiodic); the other
// categories fall back to brute force and are counted as "unimplemented".
#include "array2d.hpp"
#include "witness.hpp"
#include "oracle.hpp"
#include "duel.hpp"
#include "instrument.hpp"
#include "classify.hpp"
#include "case_lattice.hpp"
#include "case_line.hpp"
#include <optional>
#include <string>
#include <algorithm>

namespace gp {

struct StageInfo {
    View P;
    Classification cls;
};

struct Result {
    WitnessTable table;
    std::vector<StageInfo> stages;   // stages.back() describes P itself
    Result(int cap) : table(cap) {}
    Category category() const { return stages.back().cls.cat; }
};

// ---- Section 4.1: P^{t-1} nonperiodic ---------------------------------------------------
// Every vector of C^{t-1} has a witness, so any two vectors of one new sub-box can duel.
template <class T>
void case_nonperiodic(const Array2D<T>& A, const View& cur, const View& prev,
                      WitnessTable& tab, int K, int Kp) {
    gp_path("4.1");
    for (bool quad_one : {true, false}) {
        for (int k = 1; k <= 4; ++k) {
            if (quad_one ? (k == 1) : (k == 2)) continue;          // that sub-box is C^{t-1}
            std::vector<Vec2> cands = WitnessTable::subbox(quad_one, k, K, Kp);
            std::optional<Vec2> surv;
            for (Vec2 v : cands) {
                if (tab.at(v).s == Status::Witness) continue;
                if (!surv) { surv = v; continue; }
                PairOutcome o = duel_pair(A, cur, tab, *surv, v, Rule::Any, &prev);
                if (!gp_check(o.ok, "4.1:duel")) {                 // fallback: settle v directly
                    tab.at(v) = brute_force_witness(A, cur, v);
                    if (tab.at(v).s == Status::Period) { tab.at(*surv) = brute_force_witness(A, cur, *surv); surv = v; }
                    continue;
                }
                if (o.wx) tab.set_witness(*surv, *o.wx);
                if (o.wy) tab.set_witness(v, *o.wy);
                if (o.wx && !o.wy) surv = v;
                else if (o.wx && o.wy) surv.reset();
                // else surv keeps, v lost
            }
            if (surv) tab.at(*surv) = brute_force_witness(A, cur, *surv);
        }
    }
}

// ---- driver ---------------------------------------------------------------------------
template <class T>
Result compute_witnesses(const Array2D<T>& A, int base = 16) {
    int m = A.n;
    std::vector<View> chain;
    View V{0, 0, m};
    chain.push_back(V);
    while (V.size > base) { V = V.inner(); chain.push_back(V); }
    std::reverse(chain.begin(), chain.end());

    Result res(View{0, 0, m}.K());
    WitnessTable& tab = res.table;

    // base stage: brute force
    gp_path("base");
    brute_force_box(A, chain[0], tab, chain[0].K());
    res.stages.push_back({chain[0], classify(tab, chain[0].K())});

    for (std::size_t t = 1; t < chain.size(); ++t) {
        const View& prev = chain[t - 1];
        const View& cur = chain[t];
        int Kp = prev.K(), K = cur.K();
        const Classification& pc = res.stages.back().cls;
        switch (pc.cat) {
            case Category::Nonperiodic:
                case_nonperiodic(A, cur, prev, tab, K, Kp);
                break;
            case Category::Lattice:
                case_lattice(A, cur, prev, tab, K, Kp, *pc.vI, *pc.vII);
                break;
            case Category::LineI: case Category::LineII: case Category::RadiantI: case Category::RadiantII:
                case_line(A, cur, prev, tab, K, Kp, pc);
                break;
            default:
                gp_unimplemented(category_name(pc.cat));
                // Witness entries stay valid; everything else (new vectors and inherited
                // periods) is recomputed against P^t.
                for (Vec2 v : WitnessTable::boxI(K)) if (tab.at(v).s != Status::Witness) tab.at(v) = brute_force_witness(A, cur, v);
                for (Vec2 v : WitnessTable::boxII(K)) if (tab.at(v).s != Status::Witness) tab.at(v) = brute_force_witness(A, cur, v);
                break;
        }
        // safety sweep: nothing may remain Unknown
        for (bool q1 : {true, false})
            for (Vec2 v : (q1 ? WitnessTable::boxI(K) : WitnessTable::boxII(K)))
                if (tab.at(v).s == Status::Unknown) {
                    gp_check(false, "sweep:unknown");
                    tab.at(v) = brute_force_witness(A, cur, v);
                }
        res.stages.push_back({cur, classify(tab, K)});
    }
    return res;
}

} // namespace gp
