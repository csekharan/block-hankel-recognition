#pragma once
// Direct comparison baseline: for every non-trivial admissible pair (p,q), compare T_{p,q} and
// B_{p,q} in place, row segment by row segment (memcmp), stopping at the first mismatch.
// No preprocessing. Worst case O(mn d(m) d(n)); early exit makes rejected pairs cheap unless the
// matrix is nearly block Hankel.
#include <cstring>
#include <set>
#include <utility>
#include "../gp/array2d.hpp"
#include "paper_hash.hpp"   // candidate_pairs

namespace bh {
template <class T>
inline bool direct_pair(const gp::Array2D<T>& A, int p, int q) {
    const int m = A.n, n = A.n;
    const std::size_t bytes = (std::size_t)(n - q) * sizeof(T);
    for (int x = 0; x < m - p; ++x) {
        const T* top = &A.a[(std::size_t)x * n + q];          // row x of T_{p,q}
        const T* bot = &A.a[(std::size_t)(x + p) * n];        // row x of B_{p,q}
        if (std::memcmp(top, bot, bytes) != 0) return false;
    }
    return true;
}
template <class T>
inline std::set<std::pair<int,int>> direct_pairs(const gp::Array2D<T>& A) {
    std::set<std::pair<int,int>> out;
    for (PQ pq : candidate_pairs(A.n, A.n)) if (direct_pair(A, pq.p, pq.q)) out.insert({pq.p, pq.q});
    return out;
}
}
