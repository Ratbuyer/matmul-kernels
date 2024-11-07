// kernel 3, 2d titling

constexpr int b_M = 64;
constexpr int b_N = 8;
constexpr int b_K = 16;

constexpr int w_M = 16;
constexpr int w_N = 8;
constexpr int w_K = 16;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;

__global__ void kernel_3(half *A, half *B, half* C, int M, int N, int K) {
	
	const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Block and thread indices
    int blockRow = blockIdx.x / (N / b_N);
    int blockCol = blockIdx.x % (N / b_N);
    
    __shared__ half As[b_M * b_K];
    __shared__ half Bs[b_K * b_N];
    
    int4 *As_int4 = reinterpret_cast<int4 *>(As + warpId * w_M * w_K);
    
    // pointers for vectorized loading
    int4 *A_int4 = reinterpret_cast<int4 *>(A + (blockRow * b_M + warpId * w_M) * K);
    
    half acc[4] = {0};
    
    for (int k = 0; k < K / b_K; k++) {
		// each thread loads 1 elements from B
		int t_row = threadIdx.x / 8;
		int t_col = threadIdx.x % 8;
		Bs[t_row * 8 + t_col] = B[(k * b_K + t_row) * N + blockCol * b_N + t_col];
		
		// each thread loads 8 elements from A
    	t_row = laneId / 2;
		t_col = laneId % 2;
		As_int4[t_row * 2 + t_col] = A_int4[t_row * (K / 8) + k * 2 + t_col];
		
		// compute, each threads computes 4 elements
		__syncthreads();
		
		t_row = threadIdx.x / 2;
		t_col = threadIdx.x % 2;
		
		#pragma unroll
		for (int i = 0; i < w_K; i++) {
			acc[0] += As[t_row * w_K + i] * Bs[i * b_N + t_col * 4];
			acc[1] += As[t_row * w_K + i] * Bs[i * b_N + t_col * 4 + 1];
			acc[2] += As[t_row * w_K + i] * Bs[i * b_N + t_col * 4 + 2];
			acc[3] += As[t_row * w_K + i] * Bs[i * b_N + t_col * 4 + 3];
		}
		
    }
    
    int c_row = threadIdx.x / 2;
	int c_col = threadIdx.x % 2;
	
	__syncthreads();
    
    // store
    C[(blockRow * b_M + c_row) * N + blockCol * b_N + c_col * 4] = acc[0];
    C[(blockRow * b_M + c_row) * N + blockCol * b_N + c_col * 4 + 1] = acc[1];
    C[(blockRow * b_M + c_row) * N + blockCol * b_N + c_col * 4 + 2] = acc[2];
    C[(blockRow * b_M + c_row) * N + blockCol * b_N + c_col * 4 + 3] = acc[3];
}


void launch_kernel_3(half *A, half *B, half *C, int M, int N, int K) {
	const int BLOCKS_PER_GRID = (M / b_M) * (N / b_N);
	
	kernel_3<<<BLOCKS_PER_GRID, WARPS_PER_BLOCK * WARP_SIZE>>>(A, B, C, M, N, K);
}