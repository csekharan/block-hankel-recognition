#pragma once
#include <cstdio>
#include <string>
inline long g_checks = 0, g_fails = 0;
inline std::string g_section;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_fails; std::fprintf(stderr, "[%s] CHECK failed %s:%d: %s\n", g_section.c_str(), __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_MSG(cond, fmt, ...) do { ++g_checks; if (!(cond)) { ++g_fails; std::fprintf(stderr, "[%s] CHECK failed %s:%d: %s  " fmt "\n", g_section.c_str(), __FILE__, __LINE__, #cond, __VA_ARGS__); } } while (0)
inline void section(const char* s) { g_section = s; std::printf("== %s\n", s); std::fflush(stdout); }
inline int summary() { std::printf("checks: %ld, failures: %ld\n", g_checks, g_fails); return g_fails ? 1 : 0; }
