SM=90a
NVCC=/usr/local/cuda-12.4/bin/nvcc
INCLUDES=-I./
OPTIMIZATION=-O3
LINKS=-lcudart -lcuda
PTX=--ptxas-options=-v

OUTPUT=run

all : 1

1 : makefile run.cu 1_naive.cuh
	$(NVCC) -arch=sm_${SM} $(OPTIMIZATION) $(INCLUDES) $(LINKS) $(PTX) run.cu -o $(OUTPUT)

clean :
	rm -f $(OUTPUT)