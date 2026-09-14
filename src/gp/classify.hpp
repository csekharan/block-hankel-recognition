#pragma once
// Periodicity category of a stage, read off the finished witness table.
#include "witness.hpp"
#include <optional>
#include <vector>

namespace gp {

enum class Category { Nonperiodic, Lattice, LineI, LineII, RadiantI, RadiantII };

inline const char* category_name(Category c) {
    switch (c) {
        case Category::Nonperiodic: return "Nonperiodic";
        case Category::Lattice: return "Lattice";
        case Category::LineI: return "LineI";
        case Category::LineII: return "LineII";
        case Category::RadiantI: return "RadiantI";
        case Category::RadiantII: return "RadiantII";
    }
    return "?";
}

struct Classification {
    Category cat = Category::Nonperiodic;
    std::optional<Vec2> vI, vII;     // shortest valid periods
    bool complete = true;            // false if some entry was Unknown
};

// Category from a finished table over C^K (paper, Section 2). Line-periodic means all valid
// periods of the periodic quadrant are collinear with the origin (not necessarily multiples).
inline Classification classify(const WitnessTable& tab, int K) {
    Classification C;
    std::vector<Vec2> PI, PII;
    for (Vec2 v : WitnessTable::boxI(K)) {
        Status s = tab.at(v).s;
        if (s == Status::Unknown) C.complete = false;
        if (s == Status::Period) PI.push_back(v);
    }
    for (Vec2 v : WitnessTable::boxII(K)) {
        Status s = tab.at(v).s;
        if (s == Status::Unknown) C.complete = false;
        if (s == Status::Period) PII.push_back(v);
    }
    if (!PI.empty()) C.vI = PI.front();      // boxes are sorted in the length-lex orders
    if (!PII.empty()) C.vII = PII.front();
    if (C.vI && C.vII) C.cat = Category::Lattice;
    else if (!C.vI && !C.vII) C.cat = Category::Nonperiodic;
    else if (C.vI) {
        bool line = true;
        for (Vec2 v : PI) if (!collinear(v, *C.vI)) { line = false; break; }
        C.cat = line ? Category::LineI : Category::RadiantI;
    } else {
        bool line = true;
        for (Vec2 v : PII) if (!collinear(v, *C.vII)) { line = false; break; }
        C.cat = line ? Category::LineII : Category::RadiantII;
    }
    return C;
}

} // namespace gp
