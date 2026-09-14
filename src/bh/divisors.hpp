#pragma once
#include <vector>
#include <algorithm>
namespace bh {
inline std::vector<int> divisors(int m) {
    std::vector<int> d;
    for (int i = 1; (long)i * i <= m; ++i)
        if (m % i == 0) { d.push_back(i); if (i != m / i) d.push_back(m / i); }
    std::sort(d.begin(), d.end());
    return d;
}
}
