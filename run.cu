#include <iostream>

#include "1_naive.cuh"

constexpr int expected_argc = 4;

int main(int argc, char **argv) {
	
	if (argc != expected_argc) {
		std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <Iterations>" << std::endl;
		return 1;
	}
	
	const int M = std::stoi(argv[1]);
	const int K = std::stoi(argv[2]);
	const int N = std::stoi(argv[3]);
	const int iterations = std::stoi(argv[4]);
}