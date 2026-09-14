#pragma once
// Type-1 / type-2 duels (paper, Section 4), rules for choosing the type, and the
// candidate-reduction loop used by 4.1, B1.1, D1.1, D1.2, B3.1.
#include "array2d.hpp"
#include "witness.hpp"
#include "instrument.hpp"
#include <optional>

namespace gp {

enum class Rule { Any, Rule1, Rule2I, Rule2II };

struct DuelOutcome {
    bool ok = false;                 // false: no legal apex, or no seed witness
    std::optional<Vec2> wu, wv;      // witnesses found against u / v
    int type = 0;
};

// Precondition: v - u is quad-I or quad-II and w is a witness against v - u,
// i.e. P[w] != P[w - v + u], with w and w - v + u inside V.
template <class T>
DuelOutcome duel(const Array2D<T>& A, const View& V, Vec2 u, Vec2 v, Vec2 w, Rule rule,
                 const View* inner = nullptr) {
    Vec2 a1 = w + u, a2 = w - v;
    bool l1 = V.contains(a1), l2 = V.contains(a2);
    if (!l1 && !l2) return {};
    int type = 1;
    if (l1 && l2) {
        switch (rule) {
            case Rule::Any: type = 1; break;
            case Rule::Rule1: {
                int d1 = V.d2(a1), d2 = V.d2(a2);
                if (d1 != d2) type = d1 < d2 ? 1 : 2;
                else if (inner) {                       // Finding 1: break ties toward P^{t-1}
                    bool i1 = inner->contains(a1), i2 = inner->contains(a2);
                    type = (i1 == i2) ? 1 : (i1 ? 1 : 2);
                } else type = 1;
                break;
            }
            case Rule::Rule2I:  type = V.dist_backdiag(a1) <= V.dist_backdiag(a2) ? 1 : 2; break;
            case Rule::Rule2II: type = V.dist_maindiag(a1) <= V.dist_maindiag(a2) ? 1 : 2; break;
        }
    } else type = l1 ? 1 : 2;
    DuelOutcome out; out.ok = true; out.type = type;
    const T& Pw = A.at(w);
    if (type == 1) {
        if (!(A.at(a1) == Pw)) out.wu = a1;   // (w+u) - u = w
        else out.wv = a1;                      // (w+u) - v = w - v + u
    } else {
        if (!(A.at(a2) == Pw)) out.wv = w;    // w - v
        else out.wu = w - v + u;               // (w-v+u) - u = w - v
    }
    return out;
}

struct PairOutcome { bool ok = false; std::optional<Vec2> wx, wy; };

// Duel between arbitrary candidates x, y using the table's witness against their difference.
template <class T>
PairOutcome duel_pair(const Array2D<T>& A, const View& V, const WitnessTable& tab,
                      Vec2 x, Vec2 y, Rule rule, const View* inner = nullptr) {
    Vec2 u, v; bool swapped;
    if (quadI_or_II(y - x)) { u = x; v = y; swapped = false; }
    else if (quadI_or_II(x - y)) { u = y; v = x; swapped = true; }
    else return {};
    auto w = tab.witness_against(v - u);
    if (!w) return {};
    DuelOutcome o = duel(A, V, u, v, *w, rule, inner);
    if (!o.ok) return {};
    PairOutcome p; p.ok = true;
    p.wx = swapped ? o.wv : o.wu;
    p.wy = swapped ? o.wu : o.wv;
    return p;
}

} // namespace gp
