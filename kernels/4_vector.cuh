#pragma once

#include "constants.cuh"

__global__ void kernel_4(half *A, half *B, half* C, int M, int N, int K) {
	
	const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Block and thread indices
    const int blockRow = blockIdx.x / (N / b_N);
    const int blockCol = blockIdx.x % (N / b_N);
    
	const int warp_row = warpId / (b_N / w_N);
	const int warp_col = warpId % (b_N / w_N);
	
	const int thread_row = laneId / (w_N / t_N);
	const int thread_col = laneId % (w_N / t_N);
    
    __shared__ __align__(16) half As[b_M * b_K];
    __shared__ __align__(16) half Bs[b_K * b_N];
    
    int4 * A_int4 = reinterpret_cast<int4 *>(A);
    int4 * B_int4 = reinterpret_cast<int4 *>(B);
    
    int4 * As_int4 = reinterpret_cast<int4 *>(As);
    int4 * Bs_int4 = reinterpret_cast<int4 *>(Bs);

    half ar[t_M];
    half br[t_N];
    half acc[t_M][t_N] = {0};
    
    const int t_load_row = threadIdx.x / 8;
    const int t_load_col = threadIdx.x % 8;

    for (int k = 0; k < K / b_K; k++) {
		// each thread loads one row of A
		#pragma unroll
		for (int a = 0; a < 4; a++) {
			As_int4[(a * 16 + t_load_row) * b_K / 8 + t_load_col] = 
				A_int4[(blockRow * b_M + a * 16 + t_load_row) * K / 8 + k * b_K / 8 + t_load_col];
		}
		
		// each thread loads one column of B
		#pragma unroll
		for (int b = 0; b < 4; b++) {
			Bs_int4[(b * 16 + t_load_row) * b_K / 8 + t_load_col] = 
				B_int4[(k * b_K + b * 16 + t_load_row) * N / 8 + blockCol * b_N / 8 + t_load_col];
		}
		
		// compute, each threads computes 16x8
		__syncthreads();
		
		#pragma unroll
		for (int wk = 0; wk < b_K; wk++) {
			// load a from shared memory to register
			#pragma unroll
			for (int i = 0; i < t_M; i++) {
				ar[i] = As[(warp_row * w_M + thread_row * t_M + i) * b_K + wk];
			}
			
			// load b from shared memory to register
			#pragma unroll
			for (int i = 0; i < t_N; i++) {
				br[i] = Bs[wk * b_N + warp_col * w_N + thread_col * t_N + i];
			}
			
			// compute
			for (int i = 0; i < t_M; i++) {
				for (int j = 0; j < t_N; j++) {
					acc[i][j] += ar[i] * br[j];
				}
			}
		}
    }

	__syncthreads();
    
    // store
    #pragma unroll
	for (int i = 0; i < t_M; i++) {
		#pragma unroll
		for (int j = 0; j < t_N; j++) {
			C[(blockRow * b_M + warp_row * w_M + thread_row * t_M + i) * N + blockCol * b_N + warp_col * w_N + thread_col * t_N + j] = acc[i][j];
		}
	}
}


void launch_kernel_4(half *A, half *B, half *C, int M, int N, int K) {
	const int BLOCKS_PER_GRID = (M / b_M) * (N / b_N);
	
	kernel_4<<<BLOCKS_PER_GRID, WARPS_PER_BLOCK * WARP_SIZE>>>(A, B, C, M, N, K);
}