#include <complex>
#include <iostream>
#include <random>
#include <vector>
#include <math.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cufftXt.h>
#include "regression.h"

using scalar_type = float;
using data_type = std::complex<scalar_type>;

extern "C" 
void hilbert(float *signal, float *signal90, float *envelope, int N)
{
    cufftHandle plan;
    int batch_size = 1;
    int fft_size = batch_size * N;

    std::vector<data_type> czeros(fft_size);
    std::vector<data_type> data(N);
    
    for (int i = 0; i < N; i++) {
        data[i] = data_type(signal[i], 0.0f);
        czeros[i] = data_type(0.0f, 0.0f);
    }
        
    cufftComplex *d_data = nullptr;

    cufftCreate(&plan);
    cufftPlan1d(&plan, N, CUFFT_C2C, batch_size);


    // Create device data arrays
    float *signal90_d, *env_d;

    cudaMallocManaged(&signal90_d, sizeof(scalar_type) * N);
    cudaMallocManaged(&env_d, sizeof(scalar_type) * N);
    cudaMallocManaged(reinterpret_cast<void **>(&d_data), sizeof(data_type) * N);
    memcpy(d_data, data.data(), sizeof(data_type) * N);

    //transformada direta
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    
    //zera frequencias negativas (w > pi/2)
    memcpy(&d_data[fft_size/2], czeros.data(), sizeof(data_type) * N/2);
    
    //transformada inversa
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    
    
    //memcpy(output, d_data, sizeof(data_type) * N);
    for (int i = 0; i < N; i++) {
        signal90_d[i] = 2*d_data[i].y/N;
        env_d[i] = 2*sqrt(d_data[i].x*d_data[i].x + d_data[i].y*d_data[i].y)/N;
    }
    
    
	memcpy(signal90, signal90_d, sizeof(scalar_type) * N);
	memcpy(envelope, env_d, sizeof(scalar_type) * N);	


    /*for (int i = 0; i < N; i++) {
    	printf("%f, ", envelope[i]);
    }*/
    
    /* free resources */
    cudaFree(d_data);
    cudaFree(signal90_d);
    cudaFree(env_d);

    cufftDestroy(plan);
    //cudaDeviceReset();
    
}

/*
int main(int argc, char *argv[]) {
    

    int n = 128;
    
    float *dados, *transformada, *env;
    dados = (float*)malloc(sizeof(float)*n);
    transformada = (float*)malloc(sizeof(float)*n);
    env = (float*)malloc(sizeof(float)*n);

    
    for (int i = 0; i < n; i++) {
        dados[i] = 5.0f*sin(3.0f*3.14159f*i/n)*sin(20.0f*2.0f*3.14159f*i/n);
    }

    std::printf("ss = [");
    for (int i = 0; i < n; i++) {
        std::printf("%f, ", dados[i]);
    }
    std::printf("];\n");

    hilbert(dados, transformada, env, n);

    std::printf("tt = [");
    for (int i = 0; i < n; i++) {
        std::printf("%f, ", transformada[i]);
    }
    std::printf("];\n");
    
    std::printf("ee = [");
    for (int i = 0; i < n; i++) {
        std::printf("%f, ", env[i]);
    }
    std::printf("];\n");

    return EXIT_SUCCESS;
}*/
