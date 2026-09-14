#pragma once
// Safety net: GP_CHECK aborts in test builds, otherwise records a fallback and lets the caller
// recover by brute force. PathCounter records which steps of the paper executed.
#include <map>
#include <string>
#include <cstdio>
#include <cstdlib>

namespace gp {

struct Counters {
    std::map<std::string, long> path, fallback, unimplemented;
    void reset() { path.clear(); fallback.clear(); unimplemented.clear(); }
    long total_fallbacks() const { long s = 0; for (auto& [k, v] : fallback) s += v; return s; }
    long total_unimplemented() const { long s = 0; for (auto& [k, v] : unimplemented) s += v; return s; }
    void print(FILE* f = stdout) const {
        std::fprintf(f, "paths:");
        for (auto& [k, v] : path) std::fprintf(f, " %s=%ld", k.c_str(), v);
        std::fprintf(f, "\nfallbacks:");
        for (auto& [k, v] : fallback) std::fprintf(f, " %s=%ld", k.c_str(), v);
        std::fprintf(f, "\nunimplemented:");
        for (auto& [k, v] : unimplemented) std::fprintf(f, " %s=%ld", k.c_str(), v);
        std::fprintf(f, "\n");
    }
};
inline Counters& counters() { static Counters c; return c; }
inline bool& abort_on_assert() { static bool b = false; return b; }

// Returns cond; on failure either aborts (tests) or records a fallback (release).
inline bool gp_check(bool cond, const char* what) {
    if (cond) return true;
    if (abort_on_assert()) { std::fprintf(stderr, "GP_ASSERT failed: %s\n", what); std::abort(); }
    counters().fallback[what]++;
    return false;
}
// Never aborts: records a fallback of the algorithmic argument and lets the caller recover.
inline bool gp_soft_check(bool cond, const char* what) {
    if (cond) return true;
    counters().fallback[what]++;
    return false;
}
inline void gp_path(const char* what) { counters().path[what]++; }
inline void gp_unimplemented(const char* what) { counters().unimplemented[what]++; }

} // namespace gp
