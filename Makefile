# Block-Hankel recognition benchmark.  GCC or Clang with 128-bit integer support (Linux, macOS,
# MinGW-w64, WSL).  MSVC is not supported (unsigned __int128).
CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O3 -march=native -DNDEBUG -Wall -Wextra
PY       ?= python3

SRC      := src/tests/compare.cpp
HDRS     := $(wildcard src/gp/*.hpp src/bh/*.hpp src/tests/*.hpp)
BIN      := compare

.PHONY: all smoke fp bench figures clean

all: $(BIN)

$(BIN): $(SRC) $(HDRS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)

# Quick correctness check: two small sizes, one repetition; must print disagreements=0.
smoke: $(BIN)
	./$(BIN) time smoke.csv 240,480 1 0

# False-positive study of the Monte Carlo recognizer on 240 small matrices (24 sizes x 10 kinds).
fp: $(BIN)
	./$(BIN) fp 240 fp.csv

# The ten-size ladder of the paper (about 25 min on a 2-vCPU Colab runtime, ~5 GB at m = 15000).
bench: $(BIN)
	./$(BIN) time time.csv 240,480,720,960,1440,1920,2880,3840,7680,15000 3 0

# Redraw all figures from results/time_colab.csv (needs pandas + matplotlib).
figures:
	$(PY) scripts/plot_results.py

clean:
	rm -f $(BIN) $(BIN).exe smoke.csv fp.csv time.csv
