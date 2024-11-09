#pragma once

#include "constants.cuh"

__global__ void kernel_6(half *A, half *B, half* C, int M, int N, int K) {
	
	const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Block and thread indices
    const int blockRow = blockIdx.x / (N / b_N);
    const int blockCol = blockIdx.x % (N / b_N);
    
    __shared__ __align__(16) half As[2][b_M * b_K];
    __shared__ __align__(16) half Bs[2][b_K * b_N];

    half ar[8];
    half br[8][4];
    half acc[8][4] = {0};
    
    uint32_t * ar_int = reinterpret_cast<uint32_t *>(ar);
    
    const int t_load_row = threadIdx.x / 8;
    const int t_load_col = threadIdx.x % 8;
    
    // for mma indexing
    const int groupId = laneId >> 2;
    const int threadId_in_group = laneId % 4;
    
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
		for (int wk = 0; wk < 4; wk++) {
			// load a to registers
			ar[0] = As[phase][(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2];
			ar[1] = As[phase][(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2 + 1];
			ar[2] = As[phase][(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2];
			ar[3] = As[phase][(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2 + 1];
			ar[4] = As[phase][(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2 + 8];
			ar[5] = As[phase][(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2 + 8 + 1];
			ar[6] = As[phase][(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2 + 8];
			ar[7] = As[phase][(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2 + 8 + 1];
			
			// load b to registers
			#pragma unroll
			for (int b = 0; b < 8; b++) {
				br[b][0] = Bs[phase][(wk * 16 + threadId_in_group * 2) * b_N + b * 8 + groupId];
				br[b][1] = Bs[phase][(wk * 16 + threadId_in_group * 2 + 1) * b_N + b * 8 + groupId];
				br[b][2] = Bs[phase][(wk * 16 + threadId_in_group * 2 + 8) * b_N + b * 8 + groupId];
				br[b][3] = Bs[phase][(wk * 16 + threadId_in_group * 2 + 8 + 1) * b_N + b * 8 + groupId];
			}
			
			// compute
			#pragma unroll
			for (int c = 0; c < 8; c++) {
				MMA_FP16_M16N8K16(reinterpret_cast<uint32_t *>(acc[c]), ar_int, reinterpret_cast<uint32_t *>(br[c]));
			}
		}
    }
    
    // compute tail
    {
    cp_async_wait_group<0>();
    const int phase = k % 2;
    
    for (int wk = 0; wk < 4; wk++) {
		// load a to registers
		ar[0] = As[phase][(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2];
		ar[1] = As[phase][(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2 + 1];
		ar[2] = As[phase][(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2];
		ar[3] = As[phase][(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2 + 1];
		ar[4] = As[phase][(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2 + 8];
		ar[5] = As[phase][(warpId * 16 + groupId) * b_K + wk * 16 + threadId_in_group * 2 + 8 + 1];
		ar[6] = As[phase][(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2 + 8];
		ar[7] = As[phase][(warpId * 16 + groupId + 8) * b_K + wk * 16 + threadId_in_group * 2 + 8 + 1];
		
		// load b to registers
		#pragma unroll
		for (int b = 0; b < 8; b++) {
			br[b][0] = Bs[phase][(wk * 16 + threadId_in_group * 2) * b_N + b * 8 + groupId];
			br[b][1] = Bs[phase][(wk * 16 + threadId_in_group * 2 + 1) * b_N + b * 8 + groupId];
			br[b][2] = Bs[phase][(wk * 16 + threadId_in_group * 2 + 8) * b_N + b * 8 + groupId];
			br[b][3] = Bs[phase][(wk * 16 + threadId_in_group * 2 + 8 + 1) * b_N + b * 8 + groupId];
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


void launch_kernel_6(half *A, half *B, half *C, int M, int N, int K) {
	const int BLOCKS_PER_GRID = (M / b_M) * (N / b_N);
	
	kernel_6<<<BLOCKS_PER_GRID, WARPS_PER_BLOCK * WARP_SIZE>>>(A, B, C, M, N, K);
}