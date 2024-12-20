SM=86
NVCC=/usr/local/cuda-12.4/bin/nvcc
INCLUDES=-I./kernels
OPTIMIZATION=-O3
LINKS=-lcudart -lcuda
PTX=--ptxas-options=-v

OUTPUT=run

all : 1

1 : makefile run.cu kernels/1_naive.cuh
	$(NVCC) -arch=sm_${SM} $(OPTIMIZATION) $(INCLUDES) $(LINKS) $(PTX) run.cu -o $(OUTPUT)

clean :
	rm -f $(OUTPUT)