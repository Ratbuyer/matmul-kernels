sm_version=90a
NVCC=/usr/local/cuda-12.4/bin/nvcc
INCLUDES=-I./headers/device/ -I./headers/host/
OPTIMIZATION=-O0
LINKS=-lcudart -lcuda

all : 1

1 : makefile run.cu 1_naive.cuh
	$(NVCC) $(OPTIMIZATION) $(INCLUDES) $(LINKS) run.cu -o run
	
clean :
	rm -f 1