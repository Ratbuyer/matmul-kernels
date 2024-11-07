// kernel 3, 2d titling

constexpr int b_M = 128;
constexpr int b_N = 8;
constexpr int b_K = 128;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;

__global__ void kernel_4(half *A, half *B, half* C, int M, int N, int K) {
	
	const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Block and thread indices
    int blockRow = blockIdx.x / (N / b_N);
    int blockCol = blockIdx.x % (N / b_N);
    
    __shared__ half As[b_M * b_K];
    __shared__ half Bs[b_K * b_N];
    
    int4 *As_int4 = reinterpret_cast<int4 *>(As);
    int4 *Bs_int4 = reinterpret_cast<int4 *>(Bs);
    
    // pointers for vectorized loading
    int4 *A_int4 = reinterpret_cast<int4 *>(A + (blockRow * b_M + threadIdx.x) * K);
    int4 *B_int4 = reinterpret_cast<int4 *>(B + blockCol * b_N);
    
    half acc[8] = {0};
    half b[8];
    
    for (int k = 0; k < K / b_K; k++) {
		// load a to shared memory
		#pragma unroll
		for (int a = 0; a < 128 / 8; a++) {
			As_int4[threadIdx.x * (b_K / 8) + a] = A_int4[k * (128 / 8) + a];
		}
		
		// load b to shared memory
		Bs_int4[threadIdx.x] = B_int4[((k * b_K) + threadIdx.x) * (N / 8)];
		
		// compute, each threads computes a row of c (8 elements)
		__syncthreads();
		
		for (int k2 = 0; k2 < b_K; k2++) {
			// load b to registers
			#pragma unroll
			for (int i = 0; i < 8; i++) {
				b[i] = Bs[k2 * b_N + i];
			}
			
			// load a to register
			half a = As[threadIdx.x * b_K + k2];
			
			#pragma unroll
			for (int i = 0; i < 8; i++) {
				acc[i] += a * b[i];
			}
		}
    }

	__syncthreads();
    
    // store
    #pragma unroll
    for (int i = 0; i < 8; i++) {
    	C[(blockRow * b_M + threadIdx.x) * N + blockCol * b_N + i] = acc[i];
    }
}


void launch_kernel_4(half *A, half *B, half *C, int M, int N, int K) {
	const int BLOCKS_PER_GRID = (M / b_M) * (N / b_N);
	
	kernel_4<<<BLOCKS_PER_GRID, WARPS_PER_BLOCK * WARP_SIZE>>>(A, B, C, M, N, K);
}