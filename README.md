# Block-Hankel recognition: four algorithms on identical matrices

[![build](https://github.com/csekharan/block-hankel-recognition/actions/workflows/build.yml/badge.svg)](https://github.com/csekharan/block-hankel-recognition/actions/workflows/build.yml)

Header-only C++20 implementations of four algorithms that find **every block size `(p, q)` for which a
square matrix is block Hankel**, plus a shared-matrix benchmark harness that runs all four on the same
inputs, cross-checks their answers, and times them on matrices from 240 × 240 to 15000 × 15000
(225 million entries).

| # | Method | Kind | Source |
|---|--------|------|--------|
| 1 | **Galil–Park** two-dimensional witness computation + front end | deterministic, exact | [`src/gp/`](src/gp/), [`src/bh/block_hankel.hpp`](src/bh/block_hankel.hpp) |
| 2 | **Row Z-pass** over every divisor `q` | deterministic, exact | [`src/bh/block_hankel.hpp`](src/bh/block_hankel.hpp) (`zpass_rows`) |
| 3 | **2-D polynomial hashing**, `k = 2` families, `P = 2^61 − 1` | Monte Carlo, one-sided error | [`src/bh/paper_hash.hpp`](src/bh/paper_hash.hpp) |
| 4 | **Direct comparison** of the overlap matrices with early exit | deterministic baseline | [`src/bh/direct.hpp`](src/bh/direct.hpp) |

Method 3 is the algorithm of C. N. Sekharan, *A Linear-Time Monte Carlo Algorithm for Recognizing Block
Hankel Matrices* (manuscript TCS-D-26-00739), implemented step by step from the paper's definitions and
propositions. Methods 1, 2 and 4 are the deterministic comparison points.

**Headline result (m = 15000, medians of 3 matrices, best of 3 runs, one Colab vCPU):**

| Method | Block Hankel positives | Random negatives | Positives, relative to hashing |
|---|---:|---:|---:|
| Galil–Park (deterministic) | 63.2 s | 24.7 s | 15.7 × |
| Row Z-pass (deterministic) | 8.8 s | 0.08 s | 2.2 × |
| **2-D polynomial hashing, k = 2** | **4.0 s** | 3.9 s | 1 |
| Direct comparison, early exit | 11.9 s | < 0.1 ms | 2.9 × |

All four methods returned identical answer sets on all 60 benchmark matrices, and the hashing recognizer
produced zero false positives in 190 512 pair tests. Full tables, per-size medians, scaling exponents and
the raw data are in [`results/RESULTS.md`](results/RESULTS.md).

---

## 1. The problem

Let `A` be an `m × m` matrix over some alphabet and let `p | m`, `q | m`. Partition `A` into `p × q`
blocks `B_{i,j}`. `A` is **`(p, q)` block Hankel** if the blocks are constant along block
anti-diagonals, `B_{i,j} = B_{i+1, j−1}`, i.e. `B_{i,j}` depends only on `i + j`. Entry-wise this is

```
A[i + p, j] = A[i, j + q]      for all 0 ≤ i < m − p,  0 ≤ j < m − q,
```

equivalently the two overlap matrices `T_{p,q} = A[0:m−p, q:m]` and `B_{p,q} = A[p:m, 0:m−q]` coincide,
equivalently the vector `(−p, q)` is a *period* of `A` in the sense of two-dimensional pattern matching.
That last equivalence is what makes the Galil–Park witness table applicable.

The task solved by every method here is: report **all non-trivial admissible pairs**
`p | m, q | m, p < m, q < m, (p, q) ≠ (1, 1)` for which `A` is `(p, q)` block Hankel. For a matrix side
with `d(m)` divisors there are `(d(m) − 1)² − 1` such pairs (360 at `m = 240`, 1520 at `m = 15000`).

## 2. The four implementations

### 2.1 Galil–Park witness computation (`src/gp/`, ≈1 300 lines)

An implementation of the alphabet-independent two-dimensional witness computation of Galil and Park
(SIAM J. Comput. 25(5), 1996). The matrix is processed through nested, centred sub-squares
(`View::inner`, base size 16). For each stage the *witness table* records, for every candidate vector `v`
with `|v| ≤ K = ⌈size/4⌉ − 1` in quadrants I and II, either "period" or a concrete position at which the
matrix disagrees with its `v`-shift. The previous stage's table is classified as nonperiodic, lattice,
line or radiant ([`classify.hpp`](src/gp/classify.hpp)) and the corresponding case of the paper is run:

| Case | File | Paper section |
|---|---|---|
| nonperiodic: duels between candidates in each new sub-box | [`stage.hpp`](src/gp/stage.hpp), [`duel.hpp`](src/gp/duel.hpp) | 4.1 |
| lattice-periodic with basis `(vI, vII)`, steps A1–A5 | [`case_lattice.hpp`](src/gp/case_lattice.hpp), [`lattice.hpp`](src/gp/lattice.hpp) | 4.2 |
| line-periodic and radiant-periodic, procedure LINE | [`case_line.hpp`](src/gp/case_line.hpp), [`line.hpp`](src/gp/line.hpp) | 4.3, 4.4 |

Every witness is verified by one symbol comparison before it is stored, so a misread lemma can only
cost a counted brute-force fallback ([`instrument.hpp`](src/gp/instrument.hpp)), never a wrong entry.
The block-Hankel **front end** `bh::find_pairs_gp` reads pairs with `p, q ≤ K` straight from the finished
table and settles the few long divisors (`m/2, m/3, m/4`) with row Z-passes on `A` and `Aᵀ`.
Running time is linear in the number of entries with a large constant; peak memory is dominated by the
witness table.

### 2.2 Row Z-pass (`bh::zpass_rows`)

For a fixed column offset `q`, treat every row as a super-symbol: `right_x = A[x, q:m]` and
`left_x = A[x, 0:m−q]`. The Z-array (longest-common-prefix array of Main and Lorentz) of the sequence
`right_0 … right_{m−1} $ left_0 … left_{m−1}`, with row equality decided by `memcmp`, yields for **all
`p` at once** whether `(−p, q)` is a period. One pass per divisor `q < m` therefore answers every pair.
The Z-algorithm performs `O(m)` row comparisons per pass; each comparison stops at the first differing
byte, so random matrices cost almost nothing, while on block-Hankel matrices whole rows are compared.

### 2.3 Two-dimensional polynomial hashing (`src/bh/paper_hash.hpp`)

The paper's Algorithm 1. With random bases `β1` (columns) and `β2` (rows) in `F_P`, `P = 2^61 − 1`,
the hash of a rectangle is `Σ A[i,j] β1^{c2−1−j} β2^{r2−1−i} mod P`. A prefix table `G[r, c]` of the hashes
of all top-left rectangles is built in one pass by the recurrences of Proposition 5; the hash of any
sub-rectangle is then four table look-ups and three multiplications (Proposition 6). A pair `(p, q)` is
accepted when `hash(T_{p,q}) = hash(B_{p,q})` for all `k` independent families (Definition 7); `k = 2` in the
benchmark. The recognizer never rejects a true pair; a false acceptance of one pair has probability at
most `((m − p) + (m − q) − 2)/(P − 1)` per family (Theorem 1), about `10^{−14}` per family at
`m = 15000` and `10^{−28}` for `k = 2`. The `k` families are built jointly, with the Horner chains of
four rows and all families interleaved so that independent multiplications overlap in the pipeline.
Total work is `O(k m²)` for the tables plus `O(k)` per candidate pair, independent of the input.

`src/bh/hash2d.hpp` contains an earlier prefix-sum formulation of the same idea (kept for reference; the
harness benchmarks `paper_hash.hpp`).

### 2.4 Direct comparison with early exit (`src/bh/direct.hpp`)

No preprocessing: for every candidate pair compare `T_{p,q}` and `B_{p,q}` row segment by row segment with
`memcmp` and stop at the first mismatch. Rejections on random input are almost free; every accepted pair
costs a full `Θ(m²)` scan, and a *near* block-Hankel matrix (all pairs rejected only in the last row)
costs `Θ(m² · d(m)²)`. This is the natural baseline against which the other three are measured.

## 3. Repository layout

```
src/gp/        Galil–Park witness computation (header-only)
src/bh/        block-Hankel front ends: GP table look-up + Z-pass, hashing, direct comparison
src/tests/     compare.cpp (benchmark + cross-check harness), gen.hpp (matrix generators)
notebooks/     block_Hankel_benchmarking.ipynb — the Colab notebook that produced results/
results/       time_colab.csv (raw), summary_medians.csv, run_log_colab.txt, RESULTS.md, figures/
scripts/       plot_results.py — regenerates every figure from the CSV
Makefile, .github/workflows/build.yml — build, smoke test and false-positive check on every push
```

## 4. Building and running

Requirements: GCC ≥ 11 or Clang with 128-bit integer support (`unsigned __int128` is used for the
modular multiplication), so Linux, macOS, WSL or MinGW-w64. MSVC is not supported.

```bash
make                       # g++ -std=c++20 -O3 -march=native -DNDEBUG src/tests/compare.cpp -o compare
make smoke                 # two small sizes, one repetition; prints "disagreements=0"
make fp                    # false-positive study of the hashing recognizer on 240 small matrices
make bench                 # the ten-size ladder of the paper (≈25 min, ≈5 GB RAM at m = 15000)
make figures               # python scripts/plot_results.py  (pandas + matplotlib)
```

The harness has two modes:

```
./compare time <csv> [sizes=240,480,960,1920,3840] [reps=3] [near=0|1]
./compare fp   <cases> <csv>
```

`time` generates six matrices per size (three block-Hankel positives with different lattice structure,
three uniform random negatives over alphabets 256, 16 and 2), runs all four methods `reps` times each,
keeps the best time, checks that the four answer sets are identical, that every planted pair was found,
and counts hashing false positives over all repetitions. `near=1` adds a seventh, near-block-Hankel
matrix per size (constant except for one entry in the last row), the worst case of the direct baseline.
`fp` runs the hashing recognizer on `cases` small matrices drawn from eight structured generators and
reports accepted pairs, false positives and the Theorem 1 bound.

To reproduce on Colab, open [`notebooks/block_Hankel_benchmarking.ipynb`](notebooks/block_Hankel_benchmarking.ipynb):
it writes the same source tree to `gpwc/`, builds it, runs the ladder and draws Figures A and B.

## 5. Benchmark design

| | |
|---|---|
| Sizes | 240, 480, 720, 960, 1440, 1920, 2880, 3840, 7680, 15000 (divisor-rich: `d(m)` = 20 … 42) |
| Matrices per size | 3 positives: random symbol on each class of the lattice spanned by two quadrant-II vectors, generators `(2,5)+(5,2)`, `(5,6)+(6,5)`, `(3,8)+(8,3)`, alphabet 8; every lattice vector `(−p, q)` with `p, q | m` is then a block-Hankel pair (2 to 22 minimal pairs per matrix plus their multiples). 3 negatives: uniform random, alphabets 256 / 16 / 2. |
| Timing | wall clock per method and matrix, best of 3 repetitions, single thread; hashing bases redrawn every repetition |
| Machine | Google Colab CPU runtime, 2 vCPUs, 12 GB RAM, g++ 11.4.0 (Ubuntu 22.04), `-std=c++20 -O3 -march=native -DNDEBUG`; the whole ladder took 1567 s |
| Correctness | four answer sets compared on every matrix (0 disagreements), planted generator pairs verified, hashing false positives counted (0 in 60 × 3 runs) |

## 6. Results

### 6.1 Charts from the notebook (linear axes)

**Figure A** — Galil–Park alone, positives (left) and negatives (right). Faded dots are the three
individual matrices; the line joins the medians.

![Figure A: Galil–Park runtimes, linear axes](results/figures/fig_A_galil_park_linear.png)

**Figure B** — the three fast methods on the same axes. On block-Hankel positives hashing is the lowest
line from about 2 million entries onward; on random negatives the two early-exit methods sit on the
x-axis while hashing does the same work as on positives.

![Figure B: Z-pass, hashing and direct comparison, linear axes](results/figures/fig_B_fast_methods_linear.png)

### 6.2 Additional views of the same data

**Figure C** — all four methods on log–log axes, which is the only way to see the nine smaller sizes.
Hashing is a straight slope-1 line in both panels; Galil–Park is parallel to it, 15 × higher. Z-pass and
direct comparison change slope with the input: near-flat on random matrices (early exit), steeper than
linear on block-Hankel matrices.

![Figure C: runtime, log-log](results/figures/fig_C_runtime_loglog.png)

**Figure D** — nanoseconds per matrix entry (median time ÷ m²). A flat line is linear scaling; the
height is the constant. Hashing costs 16–19 ns per entry regardless of input or size. Z-pass on
positives climbs from 7 to 39 ns per entry as rows get longer; on negatives it drops to 0.3 ns per
entry. Galil–Park drifts upward from about 50 to 110 ns per entry on negatives and 170 to 280 ns per
entry on positives.

![Figure D: cost per entry](results/figures/fig_D_ns_per_entry.png)

**Figure E** — the block-Hankel positives, normalised to hashing. Below 1 million entries the direct
baseline is the fastest (no preprocessing, few pairs); at 1440 the three fast methods meet; from there
on hashing pulls away, and at `m = 15000` Z-pass takes 2.2 × and direct comparison 2.9 × its time.

![Figure E: ratio to hashing on positives](results/figures/fig_E_ratio_to_hashing.png)

### 6.3 Medians per size (ms)

Block-Hankel positives:

| m | entries | Galil–Park | Z-pass | hashing k = 2 | direct |
|---:|---:|---:|---:|---:|---:|
| 240 | 57 600 | 9.7 | 0.4 | 0.9 | 0.1 |
| 480 | 230 400 | 42.6 | 1.7 | 3.7 | 0.8 |
| 720 | 518 400 | 150.9 | 6.9 | 13.9 | 5.7 |
| 960 | 921 600 | 220.1 | 10.1 | 15.2 | 8.1 |
| 1440 | 2.07 M | 426.3 | 34.3 | 32.6 | 36.6 |
| 1920 | 3.69 M | 788.4 | 78.4 | 58.1 | 77.6 |
| 2880 | 8.29 M | 1 839.5 | 277.3 | 142.0 | 492.7 |
| 3840 | 14.7 M | 3 286.6 | 429.3 | 250.8 | 508.4 |
| 7680 | 59.0 M | 15 699.4 | 2 101.4 | 1 113.9 | 2 797.0 |
| 15000 | 225 M | 63 186.6 | 8 820.8 | 4 022.6 | 11 863.1 |

Random negatives:

| m | entries | Galil–Park | Z-pass | hashing k = 2 | direct |
|---:|---:|---:|---:|---:|---:|
| 240 | 57 600 | 3.0 | 0.1 | 0.9 | 0.0 |
| 480 | 230 400 | 13.2 | 0.3 | 3.7 | 0.0 |
| 720 | 518 400 | 45.3 | 1.3 | 14.0 | 0.1 |
| 960 | 921 600 | 58.8 | 1.0 | 14.6 | 0.0 |
| 1440 | 2.07 M | 138.3 | 2.2 | 32.2 | 0.0 |
| 1920 | 3.69 M | 273.5 | 3.7 | 57.3 | 0.0 |
| 2880 | 8.29 M | 634.0 | 7.0 | 140.4 | 0.1 |
| 3840 | 14.7 M | 1 182.8 | 9.0 | 252.2 | 0.0 |
| 7680 | 59.0 M | 5 731.2 | 29.6 | 1 102.7 | 0.1 |
| 15000 | 225 M | 24 655.3 | 77.1 | 3 902.0 | 0.1 |

### 6.4 What the comparison shows

- **Hashing is the only method whose cost does not depend on the input.** Positives and negatives cost
  the same to within 3 %, and the per-entry cost is flat across three orders of magnitude of matrix size.
  This is the practical content of the paper's linear-time claim.
- **The deterministic Z-pass is competitive, and wins on negatives.** Its worst case is the positive
  input, where each `memcmp` runs to the end of the row; there it ends 2.2 × slower than hashing at the
  largest size, and its fitted exponent over the last six sizes is 1.22 rather than 1.03.
- **Direct comparison is the right tool for small matrices and for rejection.** Up to about
  1 million entries it beats everything; on random input it never leaves the first row. Its cost on
  positives grows with the number of accepted pairs (the 15000 × 15000 matrix with 22 minimal pairs took
  18.7 s against 11–12 s for the other two positives), and the `near = 1` option exposes its
  `Θ(m² d(m)²)` worst case.
- **Galil–Park is exact and linear but pays a 6–16 × constant** over hashing, because it computes
  witnesses for *every* vector up to length `m/4`, not only the `d(m)²` divisor pairs. Its value here is
  as an independent, deterministic oracle for the other three methods.
- **Agreement.** Across the 60 matrices and 180 hashing runs the four methods never disagreed, no planted
  pair was missed, and no false positive appeared (the Theorem 1 bound is about `10^{−28}` per pair at
  `k = 2`).

A detailed discussion, the fitted scaling exponents and the complete 60-row raw table are in
[`results/RESULTS.md`](results/RESULTS.md).

## 7. Reproducibility notes

- Every matrix is generated from a seed derived from `(m, index)`, and the hashing bases from
  `(m, index, k, repetition)`, so a rerun on the same compiler produces the same inputs.
- `results/time_colab.csv` holds the run recorded in the notebook. It was rebuilt line by line from the
  harness's console table in `results/run_log_colab.txt`; both are written by the same `printf` format,
  so the values are identical to the CSV the harness wrote on Colab.
- Figures A and B are the notebook's own PNG output (170 dpi). `scripts/plot_results.py` regenerates
  them with the notebook's code and adds Figures C–E; `results/summary_medians.csv` is its numeric output.
- At `m = 720` all four methods, including the direct baseline, show a per-entry cost 1.6–1.8 × that of
  the neighbouring sizes. Because it hits every method equally it is almost certainly a transient
  slowdown of the shared Colab VM rather than an algorithmic effect.

## 8. References

- C. N. Sekharan, *A Linear-Time Monte Carlo Algorithm for Recognizing Block Hankel Matrices*,
  manuscript TCS-D-26-00739. Definitions 4–7, Propositions 3–6, Algorithm 1 and Theorems 1–2 are
  implemented in `src/bh/paper_hash.hpp`.
- Z. Galil and K. Park, *Alphabet-independent two-dimensional witness computation*, SIAM Journal on
  Computing 25(5):907–935, 1996.
- M. G. Main and R. J. Lorentz, *An O(n log n) algorithm for finding all repetitions in a string*,
  Journal of Algorithms 5(3):422–432, 1984 (the longest-prefix array used by the Z-pass and by procedure LINE).

## License

MIT, see [LICENSE](LICENSE).
