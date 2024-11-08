#pragma once

#include "constants.cuh"

__global__ void kernel_3(half *A, half *B, half* C, int M, int N, int K) {
	
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

    half acc[t_M][t_N] = {0};
    
    half ar[t_M];
    half br[t_N];
    
    for (int k = 0; k < K / b_K; k++) {
		// each thread loads one row of A
		#pragma unroll
		for (int a = 0; a < 32; a++) {
			As[(warp_row * w_M + a) * b_K + warp_col * 32 + laneId] = 
				A[(blockRow * b_M + warp_row * w_M + a) * K + k * b_K + warp_col * 32 + laneId];
		}
		
		// each thread loads one column of B
		#pragma unroll
		for (int b = 0; b < 32; b++) {
			Bs[(warp_row * w_M + b) * b_N + warp_col * 32 + laneId] = 
				B[(k * b_K + warp_row * w_M + b) * N + blockCol * b_N + warp_col * 32 + laneId];
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
			#pragma unroll
			for (int i = 0; i < t_M; i++) {
				#pragma unroll
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
			C[(blockRow * b_M + warp_row * w_M + thread_row * t_M + i) * N + \
				blockCol * b_N + warp_col * w_N + thread_col * t_N + j] = 
				acc[i][j];
		}
	}
}


void launch_kernel_3(half *A, half *B, half *C, int M, int N, int K) {
	const int BLOCKS_PER_GRID = (M / b_M) * (N / b_N);
	
	kernel_3<<<BLOCKS_PER_GRID, WARPS_PER_BLOCK * WARP_SIZE>>>(A, B, C, M, N, K);
}