#pragma once

#include "constants.cuh"

__global__ void kernel_5(half *A, half *B, half* C, int M, int N, int K) {
	
	const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Block and thread indices
    const int blockRow = blockIdx.x / (N / b_N);
    const int blockCol = blockIdx.x % (N / b_N);
    
   	const int warp_row = warpId / (b_N / w_N);
	const int warp_col = warpId % (b_N / w_N);
	
	const int thread_row = laneId / (w_N / t_N);
	const int thread_col = laneId % (w_N / t_N);
    
    __shared__ __align__(16) half As[2][b_M * b_K];
    __shared__ __align__(16) half Bs[2][b_K * b_N];

    half ar[t_M];
    half br[t_N];
    half acc[t_M][t_N] = {0};
    
    const int t_load_row = threadIdx.x / 8;
    const int t_load_col = threadIdx.x % 8;
    
    int k = 0;
    
	for (int a = 0; a < 4; a++) {
		cp_async<16>(&As[1][(a * 16 + t_load_row) * b_K + t_load_col * 8],
					 &A[(blockRow * b_M + a * 16 + t_load_row) * K + k * b_K + t_load_col * 8]
					);
	}
	
	for (int b = 0; b < 4; b++) {
		cp_async<16>(&Bs[1][(b * 16 + t_load_row) * b_N + t_load_col * 8],
					 &B[(k * b_K + b * 16 + t_load_row) * N + blockCol * b_N + t_load_col * 8]
					);
	}
	
	cp_async_group_commit();

    for (k = 1; k < K / b_K; k++) {
		const int phase = k % 2;
		
		// each thread loads one row of A
		#pragma unroll
		for (int a = 0; a < 4; a++) {
			cp_async<16>(&As[1 - phase][(a * 16 + t_load_row) * b_K + t_load_col * 8],
						 &A[(blockRow * b_M + a * 16 + t_load_row) * K + k * b_K + t_load_col * 8]
						);
		}
		
		// each thread loads one column of B
		#pragma unroll
		for (int b = 0; b < 4; b++) {
			cp_async<16>(&Bs[1 - phase][(b * 16 + t_load_row) * b_N + t_load_col * 8],
						 &B[(k * b_K + b * 16 + t_load_row) * N + blockCol * b_N + t_load_col * 8]
						);
		}
		
		cp_async_group_commit();
		
		cp_async_wait_group<1>();
		
		// compute, each threads computes 16x8
		__syncthreads();
		
		#pragma unroll
		for (int wk = 0; wk < b_K; wk++) {
			// load a from shared memory to register
			#pragma unroll
			for (int i = 0; i < t_M; i++) {
				ar[i] = As[phase][(warp_row * w_M + thread_row * t_M + i) * b_K + wk];
			}
			
			// load b from shared memory to register
			#pragma unroll
			for (int i = 0; i < t_N; i++) {
				br[i] = Bs[phase][wk * b_N + warp_col * w_N + thread_col * t_N + i];
			}
			
			// compute
			for (int i = 0; i < t_M; i++) {
				for (int j = 0; j < t_N; j++) {
					acc[i][j] += ar[i] * br[j];
				}
			}
		}
    }
    
    // compute tail
    {
    cp_async_wait_group<0>();
    const int phase = k % 2;
    
    #pragma unroll
	for (int wk = 0; wk < b_K; wk++) {
		// load a from shared memory to register
		#pragma unroll
		for (int i = 0; i < t_M; i++) {
			ar[i] = As[phase][(warp_row * w_M + thread_row * t_M + i) * b_K + wk];
		}
		
		// load b from shared memory to register
		#pragma unroll
		for (int i = 0; i < t_N; i++) {
			br[i] = Bs[phase][wk * b_N + warp_col * w_N + thread_col * t_N + i];
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


void launch_kernel_5(half *A, half *B, half *C, int M, int N, int K) {
	const int BLOCKS_PER_GRID = (M / b_M) * (N / b_N);
	
	kernel_5<<<BLOCKS_PER_GRID, WARPS_PER_BLOCK * WARP_SIZE>>>(A, B, C, M, N, K);
}