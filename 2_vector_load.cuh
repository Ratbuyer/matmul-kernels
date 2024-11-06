// kernel 2, vectorized loading

__global__ void kernel_2_vector_load(half *A, half *B, half* C, int M, int N, int K) {
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	int row_A = tid / N;
	int col_B = tid % N;
	
	// vectorized pointer
	int4 *A_int4 = reinterpret_cast<int4*>(A + row_A * K);
	
	constexpr int HALFS_PER_INT4 = 8;
	
	half acc = 0;
	
	half A_buffer[HALFS_PER_INT4];
	int4 *A_buffer_ptr = reinterpret_cast<int4 *>(A_buffer);
	
	for (int k = 0; k < K / HALFS_PER_INT4; k++) {
		A_buffer_ptr[0] = A_int4[k];
		
		for (int i = 0; i < HALFS_PER_INT4; i++) {
			acc += A_buffer[i] * B[(k * HALFS_PER_INT4 + i) * N + col_B];
		}
	}
	
	C[row_A * N + col_B] = acc;
}

void launch_kernel_2(half *A, half *B, half *C, int M, int N, int K) {
	
	constexpr int WARP_SIZE = 32;
	constexpr int WARPS_PER_BLOCK = 4;
	
	assert((M * N) % (WARPS_PER_BLOCK * WARP_SIZE) == 0);
	
	const int BLOCKS_PER_GRID = (M * N) / (WARPS_PER_BLOCK * WARP_SIZE);
	
	kernel_2_vector_load<<<BLOCKS_PER_GRID, WARPS_PER_BLOCK * WARP_SIZE>>>(A, B, C, M, N, K);
}