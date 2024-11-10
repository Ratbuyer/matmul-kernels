#pragma once

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;

constexpr int b_M = 64;
constexpr int b_N = 64;
constexpr int b_K = 64;

constexpr int w_M = 32;
constexpr int w_N = 32;

constexpr int t_M = 8;
constexpr int t_N = 4;