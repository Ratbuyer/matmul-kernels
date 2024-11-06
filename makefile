SM=86
NVCC=/usr/local/cuda-12.4/bin/nvcc
INCLUDES=-I./
OPTIMIZATION=-O3
LINKS=-lcudart -lcuda

all : 1

1 : makefile run.cu 1_naive.cuh
	$(NVCC) -arch=sm_${SM} $(OPTIMIZATION) $(INCLUDES) $(LINKS) run.cu -o run
	

run:
	./run 4096 4096 4096 100

clean :
	rm -f run