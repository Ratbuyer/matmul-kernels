// Naive kernel, each thread processes one element of the output matrix

__global__ void kernel_1_naive(half *A, half *B, half* C, int M, int N, int K) {
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	int row_A = tid / N;
	int col_B = tid % N;
	
	half acc = 0;
	
	for (int k = 0; k < K; k++) {
		acc += __hmul(A[row_A * K + k], B[k * N + col_B]);
	}
	
	C[row_A * N + col_B] = acc;
}

void launch_kernel_1(half *A, half *B, half *C, int M, int N, int K) {
	
	constexpr int WARP_SIZE = 32;
	constexpr int WARPS_PER_BLOCK = 4;
	
	assert((M * N) % (WARPS_PER_BLOCK * WARP_SIZE) == 0);
	
	const int BLOCKS_PER_GRID = (M * N) / (WARPS_PER_BLOCK * WARP_SIZE);
	
	kernel_1_naive<<<BLOCKS_PER_GRID, WARPS_PER_BLOCK * WARP_SIZE>>>(A, B, C, M, K, N);
}

