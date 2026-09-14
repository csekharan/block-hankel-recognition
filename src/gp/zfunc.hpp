#pragma once
// Generic Z-array (the longest-prefix array LP of Main-Lorentz, Algorithm 1) over positions
// 0..n-1 with an equality functor eq(i, j). O(n) calls of eq.
#include <vector>

namespace gp {

template <class Eq>
std::vector<int> zfunc(int n, Eq eq) {
    std::vector<int> z(n, 0);
    if (n == 0) return z;
    z[0] = n;
    int l = 0, r = 0;
    for (int i = 1; i < n; ++i) {
        if (i < r) z[i] = std::min(r - i, z[i - l]);
        while (i + z[i] < n && eq(z[i], i + z[i])) ++z[i];
        if (i + z[i] > r) { l = i; r = i + z[i]; }
    }
    return z;
}

} // namespace gp
