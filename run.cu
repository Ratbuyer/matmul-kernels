#include <iostream>
#include <cassert>

#include "tools.cuh"
#include "1_naive.cuh"
#include "2_tiling.cuh"
#include "3_coalesce.cuh"
#include "4_vector.cuh"
#include "5_double_buffer.cuh"

constexpr int expected_argc = 7;

int main(int argc, char **argv) {
	
	if (argc != expected_argc) {
		std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <Iterations> <check>" << std::endl;
		return 1;
	}
	
	const int M = std::stoi(argv[1]);
	const int N = std::stoi(argv[2]);
	const int K = std::stoi(argv[3]);
	const int iterations = std::stoi(argv[4]);
	const int kernel = std::stoi(argv[5]);
	const int check = std::stoi(argv[6]);
	
	if (M <= 0 || N <= 0 || K <= 0 || iterations <= 0 || kernel <= 0 || kernel > 6 || check < 0 || check > 1) {
		std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <Iterations> <check>" << std::endl;
		return 1;
	}
	
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

	// time kernel
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;
	
	cudaEventRecord(start);
	
	for (int i = 0; i < iterations; i++) {
		switch (kernel) {
			case 1:
				launch_kernel_1(d_A, d_B, d_C, M, N, K);
				break;
			case 2:
				launch_kernel_2(d_A, d_B, d_C, M, N, K);
				break;
			case 3:
				launch_kernel_3(d_A, d_B, d_C, M, N, K);
				break;
			case 4:
				launch_kernel_4(d_A, d_B, d_C, M, N, K);
				break;
			case 5:
				launch_kernel_5(d_A, d_B, d_C, M, N, K);
				break;
			default:
				std::cerr << "Invalid kernel" << std::endl;
				return 1;
		}
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	
	// check errors
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(error)
            		<< " (Error code: " << static_cast<int>(error) << ")" << std::endl;
		return 1;
	}
	
	// printf("Latency: %f ms\n", elapsed_time / iterations);
	
	long long throughput = static_cast<long long>(M) * N * K * 2 * iterations;
    printf("Kernel: %d, M/N/K: %d, %d, %d, Throughput: %f GFLOPs/s\n", kernel, M, N, K, throughput * 1.0 / (elapsed_time * 1e6));
	
	// copy result back
	cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
	
	// print_matrix(h_C, M, N);
	
	half *cpu_C = new half[M * N];
	
	if (check) {
		CPU_gemm(h_A, h_B, cpu_C, M, N, K);
		compare_matrices(h_C, cpu_C, M, N);
	}
	
	// use this for debugging
	// print_differnce(h_C, cpu_C, M, N, 0.0);
	
	// free
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	delete[] cpu_C;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}