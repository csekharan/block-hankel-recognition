// Shared-matrix comparison harness: every test matrix is generated once and given to
//   (1) Galil-Park witness computation + front end        (deterministic)
//   (2) the row Z-pass over every divisor q               (deterministic)
//   (3) the paper's Monte Carlo recognizer, raw answers, P = 2^61 - 1, k = 1, 2, 3.
// Modes:  compare fp   <cases> <csv>   -- many small/medium matrices, false-positive counting
//         compare time <csv>           -- large matrices, timing of all methods
#include "gen.hpp"
#include "../bh/block_hankel.hpp"
#include "../bh/paper_hash.hpp"
#include "../bh/direct.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace gp;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
static bh::RectView<int> view(const Array2D<int>& A) { return {A.a.data(), A.n, A.n, (std::size_t)A.n}; }

struct Strength { std::uint64_t P; int k; const char* name; };
static const std::vector<Strength> LADDER = {       // P = 2^61 - 1, k = 2 only
    {bh::Field::M61, 2, "hash"},
};

// deterministic answer sets (nontrivial pairs, Definition 2 convention)
static std::set<std::pair<int,int>> det_zpass(const Array2D<int>& A) {
    std::set<std::pair<int,int>> out; int m = A.n; auto d = bh::divisors(m);
    for (int q : d) { if (q == m) continue; auto res = bh::zpass_rows(A, q); for (int p : d) if (p != m && !(p == 1 && q == 1) && res[p]) out.insert({p, q}); }
    return out;
}

// Minimal pairs under common scaling: (p,q) is implied by (p/k, q/k) for k >= 2 (Lemma 1 of
// Galil-Park iterated: (-p,q) a period => (-kp,kq) a period). Note (kp, q) is NOT implied.
static std::set<std::pair<int,int>> minimal_pairs(const std::set<std::pair<int,int>>& S) {
    std::set<std::pair<int,int>> out;
    for (auto [p, q] : S) {
        bool implied = false;
        for (int k = 2; k <= p && k <= q && !implied; ++k)
            if (p % k == 0 && q % k == 0 && S.count({p / k, q / k})) implied = true;
        if (!implied) out.insert({p, q});
    }
    return out;
}
static std::string fmt_pairs(const std::set<std::pair<int,int>>& S) {
    std::string o; for (auto [p, q] : S) { if (!o.empty()) o += "|"; o += std::to_string(p) + "x" + std::to_string(q); } return o.empty() ? "-" : o;
}

static Vec2 rand_quadI(std::mt19937_64& rng, int K) { return {(int)(rng() % (K + 1)), (int)(rng() % K) + 1}; }
static Vec2 rand_quadII(std::mt19937_64& rng, int K) { return {-(int)(rng() % K) - 1, (int)(rng() % (K + 1))}; }

// case id -> matrix (deterministic)
static Array2D<int> make_case(std::uint64_t id, int m, int& gen_kind) {
    std::mt19937_64 rng(id * 0x9E3779B97F4A7C15ULL + 17);
    int K = View{0, 0, m}.K(), Kp = gen::top_inner(m).K(); if (Kp < 1) Kp = 1;
    auto d = bh::divisors(m);
    gen_kind = (int)(id % 8);
    switch (gen_kind) {
        case 0: return gen::random_array(m, 2, rng());
        case 1: return gen::random_array(m, 3, rng());
        case 2: { int p = d[rng() % d.size()], q = d[rng() % d.size()]; return gen::hankel_blocks(m, p, q, 2 + (int)(rng() % 3), rng()); }
        case 3: { int p = d[rng() % d.size()], q = d[rng() % d.size()]; Array2D<int> A = gen::hankel_blocks(m, p, q, 3, rng()); A.at((int)(rng() % m), (int)(rng() % m)) ^= 1; return A; }
        case 4: return gen::lattice_with_point_defects(m, rand_quadI(rng, Kp), rand_quadII(rng, Kp), 3, (int)(rng() % 6), 0x1eu, rng());
        case 5: return gen::line_with_point_defects(m, (rng() % 2) ? rand_quadI(rng, Kp) : rand_quadII(rng, Kp), 3, (int)(rng() % 4), rng());
        case 6: { Vec2 v1, v2; if (K > Kp + 1) return gen::lattice_long_second(m, rng, rng() % 2, v1, v2); return gen::lattice(m, rand_quadI(rng, Kp), rand_quadII(rng, Kp), 2, rng()); }
        default: return gen::radiant_stage(m, rand_quadI(rng, Kp), rand_quadII(rng, Kp), 3, rng() % 2, (int)(rng() % 3), rng());
    }
}

static int fp_mode(int cases, const char* csv) {
    static const int SIZES[] = {24, 30, 32, 36, 40, 48, 60, 64, 72, 80, 90, 96, 100, 108, 120, 128, 144, 160, 180, 192, 200, 216, 240, 256};
    FILE* f = std::fopen(csv, "w");
    std::fprintf(f, "case,m,block_hankel,minimal_pairs,tested_pairs,hash_k2_accepted,false_positives,bound");
    std::fprintf(f, "\n");
    struct Agg { long fp = 0, mats_with_fp = 0, first_fp = 0; double bound = 0, bound_mat = 0; };
    std::vector<Agg> agg(LADDER.size());
    long det_mismatch = 0, false_neg = 0, total_tested = 0, total_true = 0;
    for (int c = 0; c < cases; ++c) {
        int m = SIZES[c % 24]; int kind;
        Array2D<int> A = make_case((std::uint64_t)c, m, kind);
        Result R = compute_witnesses(A);
        bh::Answer g = bh::find_pairs_gp(A, R);
        std::set<std::pair<int,int>> det = g.pairs, z = det_zpass(A);
        if (det != z) { ++det_mismatch; std::fprintf(stderr, "case %d: GP and Z-pass disagree\n", c); }
        auto pairs = bh::candidate_pairs(m, m);
        total_tested += (long)pairs.size(); total_true += (long)det.size();
        std::set<std::pair<int,int>> minimal = minimal_pairs(det);
        std::fprintf(f, "%d,%d,%s,%s,%zu", c, m, det.empty() ? "No" : "Yes", fmt_pairs(minimal).c_str(), pairs.size());
        for (std::size_t s = 0; s < LADDER.size(); ++s) {
            std::mt19937_64 rng((std::uint64_t)c * 1000 + s + 1);          // bases: reproducible per (case, strength)
            bh::Recognition rec = bh::recognize(view(A), bh::Field(LADDER[s].P), LADDER[s].k, rng);
            long fp = 0; bool first_fp = false; std::set<std::pair<int,int>> fpset;
            for (bh::PQ pq : rec.accepted) if (!det.count({pq.p, pq.q})) { ++fp; fpset.insert({pq.p, pq.q}); }
            for (auto& pq : det) { bool found = false; for (bh::PQ a : rec.accepted) if (a.p == pq.first && a.q == pq.second) found = true; if (!found) ++false_neg; }
            if (rec.first && !det.count({rec.first->p, rec.first->q})) first_fp = true;
            agg[s].fp += fp; agg[s].mats_with_fp += (fp > 0); agg[s].first_fp += first_fp;
            agg[s].bound += rec.bound_exact; agg[s].bound_mat += std::min(1.0, rec.bound_exact);
            std::fprintf(f, ",%zu,%ld,%.3e", rec.accepted.size(), fp, rec.bound_exact);
        }
        std::fprintf(f, "\n");
        if ((c + 1) % 200 == 0) { std::printf("  %d cases done\n", c + 1); std::fflush(stdout); }
    }
    std::fclose(f);
    std::printf("cases=%d  tested pairs=%ld  true pairs=%ld  GP/Z-pass mismatches=%ld  false negatives=%ld\n", cases, total_tested, total_true, det_mismatch, false_neg);
    std::printf("%-8s %12s %14s %14s %14s %12s\n", "strength", "FP pairs", "E[FP] bound", "mats w/ FP", "bound(mats)", "first=FP");
    for (std::size_t s = 0; s < LADDER.size(); ++s)
        std::printf("%-8s %12ld %14.3e %14ld %14.3e %12ld\n", LADDER[s].name, agg[s].fp, agg[s].bound, agg[s].mats_with_fp, agg[s].bound_mat, agg[s].first_fp);
    return det_mismatch || false_neg;
}

static int time_mode(const char* csv, std::vector<int> sizes, int reps, bool near) {
    // 6 matrices per size: 3 block-Hankel positives with different block shapes, 3 random negatives.
    FILE* f = std::fopen(csv, "w");
    std::fprintf(f, "size,block_hankel,minimal_pairs,gp_ms,zpass_ms,hash_k2_ms,direct_ms,false_positives\n");
    std::printf("%5s %-12s %-14s %9s %9s %9s %9s %6s\n", "size", "block_hankel", "minimal_pairs", "GP", "Zpass", "hash k2", "direct", "FP");
    // Positives with several independent block-Hankel pairs: the matrix is a random symbol
    // function on the classes of the lattice generated by two quad-II vectors (-p1,q1), (-p2,q2);
    // every lattice vector (-p,q) with p, q | m is then a block-Hankel pair.
    const int gens[3][4] = {{2, 5, 5, 2}, {5, 6, 6, 5}, {3, 8, 8, 3}};   // lattice indices 21, 11, 55
    const int alphabets[3] = {256, 16, 2};
    int disagreements = 0;
    const int nmat = near ? 7 : 6;   // optional 7th matrix: constant matrix with one entry changed (near block Hankel for every pair)
    for (int m : sizes) {           // divisor-rich sizes
        for (int idx = 0; idx < nmat; ++idx) {
            bool positive = idx < 3;
            std::mt19937_64 rng((std::uint64_t)m * 31 + idx);
            Array2D<int> A = positive ? gen::lattice(m, Vec2{-gens[idx][0], gens[idx][1]}, Vec2{-gens[idx][2], gens[idx][3]}, 8, rng())
                            : idx < 6 ? gen::random_array(m, alphabets[idx - 3], rng())
                                      : Array2D<int>(m, 7);                 // constant matrix: every admissible pair is genuine ...
            if (idx == 6) A.at(m - 1, 0) ^= 1;      // ... until one entry of the last row is changed: every pair is rejected, each only after scanning almost its whole overlap
            double best[4]; for (double& x : best) x = 1e300;
            std::set<std::pair<int,int>> det, z, h, dc; long fp = 0;
            for (int rep = 0; rep < reps; ++rep) {
                auto t0 = Clock::now(); Result R = compute_witnesses(A); bh::Answer g = bh::find_pairs_gp(A, R); auto t1 = Clock::now();
                z = det_zpass(A); auto t2 = Clock::now();
                dc = bh::direct_pairs(A); auto t3 = Clock::now();
                det = g.pairs;
                best[0] = std::min(best[0], ms(t0, t1)); best[1] = std::min(best[1], ms(t1, t2)); best[3] = std::min(best[3], ms(t2, t3));
                for (int k = 2; k <= 2; ++k) {
                    std::mt19937_64 br((std::uint64_t)m * 7 + idx * 13 + k + rep);
                    bh::Field F(bh::Field::M61);
                    auto a0 = Clock::now();
                    std::vector<std::pair<std::uint64_t, std::uint64_t>> bases; for (int r = 0; r < k; ++r) { std::uint64_t b1 = F.random_base(br), b2 = F.random_base(br); bases.push_back({b1, b2}); }
                    std::vector<bh::HashFamily<int>> fam = bh::build_families(view(A), F, bases);
                    std::set<std::pair<int,int>> acc;
                    for (bh::PQ pq : bh::candidate_pairs(m, m)) { bool all = true; for (auto& fm : fam) if (fm.hT(pq.p, pq.q) != fm.hB(pq.p, pq.q)) { all = false; break; } if (all) acc.insert({pq.p, pq.q}); }
                    auto a1 = Clock::now();
                    best[2] = std::min(best[2], ms(a0, a1));
                    h = acc;
                    for (auto& pq : acc) if (!det.count(pq)) ++fp;      // accumulated over the 3 repetitions
                }
            }
            bool agree = (det == z) && (det == dc);
            for (auto& pq : det) if (!h.count(pq)) { agree = false; std::fprintf(stderr, "false negative?!\n"); }
            if (!agree) { ++disagreements; std::fprintf(stderr, "DISAGREEMENT at size %d idx %d\n", m, idx); }
            if (positive && (!det.count({gens[idx][0], gens[idx][1]}) || !det.count({gens[idx][2], gens[idx][3]}))) { ++disagreements; std::fprintf(stderr, "positive not detected at size %d idx %d\n", m, idx); }
            if (idx == 6 && !det.empty()) { ++disagreements; std::fprintf(stderr, "near-Hankel matrix reported as block Hankel at size %d\n", m); }
            if (!positive && idx < 6 && !det.empty()) { ++disagreements; std::fprintf(stderr, "random matrix reported as block Hankel at size %d idx %d\n", m, idx); }
            std::string mp = fmt_pairs(minimal_pairs(det));          // from the deterministic answer, not the construction
            const char* kind = idx < 3 ? "Yes" : idx < 6 ? "No" : "Near";
            std::fprintf(f, "%d,%s,%s,%.1f,%.1f,%.1f,%.1f,%ld\n", m, kind, mp.c_str(), best[0], best[1], best[2], best[3], fp);
            std::printf("%5d %-12s %-14s %9.1f %9.1f %9.1f %9.1f %6ld\n", m, kind, mp.c_str(), best[0], best[1], best[2], best[3], fp);
            std::fflush(stdout);
        }
    }
    std::fclose(f);
    std::printf("disagreements=%d\n", disagreements);
    return disagreements != 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "fp") == 0) return fp_mode(argc > 2 ? std::atoi(argv[2]) : 1000, argc > 3 ? argv[3] : "fp.csv");
    if (argc >= 2 && std::strcmp(argv[1], "time") == 0) {
        // compare time <csv> [sizes=240,480,960,1920,3840] [reps=3] [near=0|1]
        std::vector<int> sizes;
        std::string sz = argc > 3 ? argv[3] : "240,480,960,1920,3840";
        for (std::size_t i = 0; i < sz.size();) { std::size_t j = sz.find(',', i); if (j == std::string::npos) j = sz.size(); sizes.push_back(std::atoi(sz.substr(i, j - i).c_str())); i = j + 1; }
        return time_mode(argc > 2 ? argv[2] : "time.csv", sizes, argc > 4 ? std::atoi(argv[4]) : 3, argc > 5 && std::atoi(argv[5]) != 0);
    }
    std::fprintf(stderr, "usage: compare fp <cases> <csv> | compare time <csv> [sizes] [reps] [near]\n");
    return 2;
}
