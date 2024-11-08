// 
#pragma once

constexpr int b_M = 128;
constexpr int b_N = 128;
constexpr int b_K = 128;

constexpr int w_M = 64;
constexpr int w_N = 64;

constexpr int t_M = 16;
constexpr int t_N = 8;
// constexpr int w_K = 16;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;

__global__ void kernel_4(half *A, half *B, half* C, int M, int N, int K) {
	
	const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Block and thread indices
    const int blockRow = blockIdx.x / (N / b_N);
    const int blockCol = blockIdx.x % (N / b_N);
    
	const int warp_row = warpId / 2;
	const int warp_col = warpId % 2;
	
	const int thread_row = laneId / 8;
	const int thread_col = laneId % 8;
    
    __shared__ __align__(16) half As[b_M * b_K];
    __shared__ __align__(16) half Bs[b_K * b_N];

    half acc[t_M][t_N] = {1};
    
    half ar[t_M];
    half br[t_N];
    
    for (int k = 0; k < K / b_K; k++) {
		// each thread loads one row of A
		#pragma unroll
		for (int a = 0; a < 128; a++) {
			As[threadIdx.x * b_K + a] = A[(blockRow * b_M + threadIdx.x) * K + k * b_K + a];
		}
		
		// each thread loads one column of B
		#pragma unroll
		for (int b = 0; b < 128; b++) {
			Bs[threadIdx.x * b_N + b] = B[(k * b_K + threadIdx.x) * N + blockCol * b_N + b];
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