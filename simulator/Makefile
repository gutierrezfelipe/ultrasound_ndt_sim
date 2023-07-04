CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I../utils
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcufft
FLAGS        := -O3 -std=c++11

all: bin/utils.o bin/hilbert.o bin/cuda_regression.so bin/cuda_regression_PML.so

bin/utils.o: src/utils.cu
	nvcc -c $(FLAGS) $(INC) src/utils.cu -o bin/utils.o

bin/hilbert.o: src/hilbert.cpp
	nvcc -c $(FLAGS) src/hilbert.cpp -o bin/hilbert.o 

bin/cuda_regression.so: src/regression.cu bin/utils.o bin/hilbert.o src/deriv_macros.h
	nvcc -shared -Xcompiler -fPIC $(FLAGS) $(INC) -o bin/cuda_regression.so src/regression.cu bin/hilbert.o bin/utils.o $(LIBS)

bin/cuda_regression_PML.so: src/regression_PML.cu bin/utils.o bin/hilbert.o src/deriv_macros.h
	nvcc -shared -Xcompiler -fPIC $(FLAGS) $(INC) -o bin/cuda_regression_PML.so src/regression_PML.cu bin/hilbert.o bin/utils.o $(LIBS)

clean:
	rm -f bin/utils.o
	rm -f bin/hilbert.o
	rm -f bin/cuda_regression.so
	rm -f bin/cuda_regression_PML.so
