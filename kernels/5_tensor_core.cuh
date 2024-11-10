#pragma once

#include "constants.cuh"

__global__ void kernel_5(half *A, half *B, half* C, int M, int N, int K) {
	
	const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Block and thread indices
    const int blockRow = blockIdx.x / (N / b_N);
    const int blockCol = blockIdx.x % (N / b_N);
    
    __shared__ __align__(16) half As[b_M * b_K];
    __shared__ __align__(16) half Bs[b_K * b_N];
    
    int4 * A_int4 = reinterpret_cast<int4 *>(A);
    int4 * B_int4 = reinterpret_cast<int4 *>(B);
    
    int4 * As_int4 = reinterpret_cast<int4 *>(As);
    int4 * Bs_int4 = reinterpret_cast<int4 *>(Bs);

    half ar[8];
    half br[8][4];
    half acc[8][4] = {0};
    
    uint32_t * ar_int = reinterpret_cast<uint32_t *>(ar);
    
    const int t_load_row = threadIdx.x / 8;
    const int t_load_col = threadIdx.x % 8;
    
    // for mma indexing
    const int groupId = laneId >> 2;
    const int threadId_in_group = laneId % 4;

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
		for (int wk = 0; wk < 4; wk++) {
			// load a to registers
			ar[0] = As[(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2];
			ar[1] = As[(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2 + 1];
			ar[2] = As[(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2];
			ar[3] = As[(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2 + 1];
			ar[4] = As[(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2 + 8];
			ar[5] = As[(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2 + 8 + 1];
			ar[6] = As[(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2 + 8];
			ar[7] = As[(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2 + 8 + 1];
			
			// load b to registers
			#pragma unroll
			for (int b = 0; b < 8; b++) {
				br[b][0] = Bs[(wk * 16 + threadId_in_group * 2) * b_N + b * 8 + groupId];
				br[b][1] = Bs[(wk * 16 + threadId_in_group * 2 + 1) * b_N + b * 8 + groupId];
				br[b][2] = Bs[(wk * 16 + threadId_in_group * 2 + 8) * b_N + b * 8 + groupId];
				br[b][3] = Bs[(wk * 16 + threadId_in_group * 2 + 8 + 1) * b_N + b * 8 + groupId];
			}
			
			// compute
			#pragma unroll
			for (int c = 0; c < 8; c++) {
				MMA_FP16_M16N8K16(reinterpret_cast<uint32_t *>(acc[c]), ar_int, reinterpret_cast<uint32_t *>(br[c]));
			}
		}
    }

	__syncthreads();
    
    // store
    #pragma unroll
    for (int n = 0; n < 8; n++) {
		C[(blockRow * b_M + warpId * 16 + groupId) * N + blockCol * b_N + n * 8 + threadId_in_group * 2] = acc[n][0];
		C[(blockRow * b_M + warpId * 16 + groupId) * N + blockCol * b_N + n * 8 + threadId_in_group * 2 + 1] = acc[n][1];
		C[(blockRow * b_M + warpId * 16 + groupId + 8) * N + blockCol * b_N + n * 8 + threadId_in_group * 2] = acc[n][2];
		C[(blockRow * b_M + warpId * 16 + groupId + 8) * N + blockCol * b_N + n * 8 + threadId_in_group * 2 + 1] = acc[n][3];
    }
}


void launch_kernel_5(half *A, half *B, half *C, int M, int N, int K) {
	const int BLOCKS_PER_GRID = (M / b_M) * (N / b_N);
	
	kernel_5<<<BLOCKS_PER_GRID, WARPS_PER_BLOCK * WARP_SIZE>>>(A, B, C, M, N, K);
}