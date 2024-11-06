#include <iostream>
#include <cassert>

#include "tools.cuh"
#include "1_naive.cuh"

constexpr int expected_argc = 5;

int main(int argc, char **argv) {
	
	if (argc != expected_argc) {
		std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <Iterations>" << std::endl;
		return 1;
	}
	
	const int M = std::stoi(argv[1]);
	const int N = std::stoi(argv[2]);
	const int K = std::stoi(argv[3]);
	const int iterations = std::stoi(argv[4]);
	
	assert(M % 16 == 0);
	assert(K % 16 == 0);
	assert(N % 16 == 0);
	
	half *h_A = new half[M * K];
	half *h_B = new half[K * N];
	half *h_C = new half[M * N];
	
	half *d_A = nullptr;
	half *d_B = nullptr;
	half *d_C = nullptr;
	
	cudaMalloc(&d_A, M * K * sizeof(half));
	cudaMalloc(&d_B, K * N * sizeof(half));
	cudaMalloc(&d_C, M * N * sizeof(half));
	
	fill_random(h_A, M, K);
	fill_random(h_B, K, N);
	
	cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
	
	constexpr int WARP_SIZE = 32;
	constexpr int WARPS_PER_BLOCK = 4;
	
	assert((M * N) % (WARPS_PER_BLOCK * WARP_SIZE) == 0);
	
	const int BLOCKS_PER_GRID = (M * N) / (WARPS_PER_BLOCK * WARP_SIZE);

	// time kernel
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;
	
	cudaEventRecord(start);
	
	for (int i = 0; i < iterations; i++) {
		kernel_1_naive<<<BLOCKS_PER_GRID, WARPS_PER_BLOCK * WARP_SIZE>>>(d_A, d_B, d_C, M, K, N);
	}
	
	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&elapsed_time, start, stop);
	
	// check errors
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
		return 1;
	}
	
	printf("Elapsed time: %f ms\n", elapsed_time);
	
	// copy result back
	cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
	
	// print_matrix(h_C, M, N);
	
	half *cpu_C = new half[M * N];
	CPU_gemm(h_A, h_B, cpu_C, M, N, K);
	
	compare_matrices(h_C, cpu_C, M, N);
}