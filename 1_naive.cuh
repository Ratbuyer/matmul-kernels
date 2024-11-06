

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

