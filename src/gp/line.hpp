#pragma once
// Procedure LINE: periods of V among the multiples of a direction v, with witnesses.
#include "array2d.hpp"
#include "witness.hpp"
#include "zfunc.hpp"

namespace gp {

// Writes into tab, for every k >= 1 with len(k v) <= K: a witness against k v if V has one,
// otherwise Period. Existing Witness entries are kept. O(size^2).
template <class T>
void line_procedure(const Array2D<T>& A, const View& V, Vec2 v, WitnessTable& tab, int K) {
    std::vector<Vec2> pts;
    for (int r = V.r0; r < V.r0 + V.size; ++r)
        for (int c = V.c0; c < V.c0 + V.size; ++c) {
            Vec2 s{r, c};
            if (V.contains(s - v)) continue;               // not the start of a line
            pts.clear();
            for (Vec2 p = s; V.contains(p); p = p + v) pts.push_back(p);
            int n = (int)pts.size();
            if (n < 2) continue;
            auto z = zfunc(n, [&](int i, int j) { return A.at(pts[i]) == A.at(pts[j]); });
            for (int i = 1; i < n; ++i) {
                Vec2 u = i * v;
                if (len(u) > K) break;
                if (i + z[i] < n && tab.status(u) != Status::Witness)
                    tab.set_witness(u, pts[i + z[i]]);   // cowitness is pts[z[i]]
            }
        }
    for (int i = 1; len(i * v) <= K; ++i)
        if (tab.status(i * v) != Status::Witness) tab.set_period(i * v);
}

} // namespace gp
