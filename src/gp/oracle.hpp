#pragma once
// Brute-force periodicity: the test oracle and the runtime fallback.
#include "array2d.hpp"
#include "witness.hpp"

namespace gp {

// O(size^2): scan all w in V with w - v in V.
template <class T>
Witness brute_force_witness(const Array2D<T>& A, const View& V, Vec2 v) {
    int r_lo = imax(V.r0, V.r0 + v.r), r_hi = imin(V.r0 + V.size - 1, V.r0 + V.size - 1 + v.r);
    int c_lo = imax(V.c0, V.c0 + v.c), c_hi = imin(V.c0 + V.size - 1, V.c0 + V.size - 1 + v.c);
    for (int r = r_lo; r <= r_hi; ++r)
        for (int c = c_lo; c <= c_hi; ++c)
            if (!(A.at(r, c) == A.at(r - v.r, c - v.c))) return {Status::Witness, {r, c}};
    return {Status::Period, {}};
}

template <class T>
void brute_force_box(const Array2D<T>& A, const View& V, WitnessTable& tab, int K) {
    for (Vec2 v : WitnessTable::boxI(K)) tab.at(v) = brute_force_witness(A, V, v);
    for (Vec2 v : WitnessTable::boxII(K)) tab.at(v) = brute_force_witness(A, V, v);
}

template <class T>
WitnessTable brute_periods(const Array2D<T>& A, const View& V) {
    WitnessTable tab(V.K());
    brute_force_box(A, V, tab, V.K());
    return tab;
}

// Validity of a stored witness with respect to view V.
template <class T>
bool check_witness(const Array2D<T>& A, const View& V, Vec2 v, Vec2 w) {
    return V.contains(w) && V.contains(w - v) && !(A.at(w) == A.at(w - v));
}

} // namespace gp
