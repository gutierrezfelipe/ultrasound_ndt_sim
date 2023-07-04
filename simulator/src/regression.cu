#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "regression.h"

#define xzt(x,z,t) ((x) + (z)*(X) + ((t)%4)*(X)*(Z))
#define xzt2(x,z,t) ((x) + (z)*(X) + ((t)%4)*(X)*(Z))
#define inbounds(x,z,offset) (((x)>=(offset) && (z)>=(offset) && (x)<(X)-(offset) && (z)<(Z)-(offset)))

//precisam ser iguais
#define WARP_SIZE (32)
#define BLOCK_SIZE (32)

#define prec_deriv (4)
#include "deriv_macros.h"


const dim3 threadGrid(BLOCK_SIZE, BLOCK_SIZE);

float *P, *cquad, *source, *record_buffer, *initial, *recording_h;
float *P_ub, *P_uf, *grad, *observed, *adj_source, *grad_h, *simulated_h, *adj_source_h;
float *P_uf_full;
int *pos_source_x, *pos_source_z, *pos_sensor_x, *pos_sensor_z;
int X, Z, T, n_source, n_sensor;
int allocated = 0;
unsigned int n_blocksX, n_blocksZ, n_blocksS, n_blocksF;
dim3 blockGrid;


__global__ void
simulateFrame (float* P, float *cquad, int X, int Z, int t)
{
    //coordenadas no bloco
    const int x = threadIdx.x;
    const int z = threadIdx.y;

    //coordenadas em Ps (shared)
    const int x_s = threadIdx.x + prec_deriv;
    const int z_s = threadIdx.y + prec_deriv;

    //coordenadas da origem do bloco
    const int x_b = blockIdx.x * blockDim.x;
    const int z_b = blockIdx.y * blockDim.y;

    //coordenadas em P (global)
    const int x_g = x_b + x;
    const int z_g = z_b + z;

    const int shared_width = BLOCK_SIZE + 2*prec_deriv;
    const int tam_shared = (shared_width)*(shared_width);

    //nao vale a pena colocar cquad e P(t-2) na memoria shared de acordo com os testes
    __shared__ float Ps[shared_width][shared_width];


    if(z==0) //first warp in block
    {
	//copy P to shared memory
	for(int id=x; id<tam_shared; id+=WARP_SIZE)
	{
	    //coordenada dentro de Ps (shared) sendo lida de P
	    const int xx = id/shared_width;
	    const int zz = id%shared_width;

	    //coordenada correspondente na memoria global
	    const int x_c = x_b - prec_deriv + xx;
	    const int z_c = z_b - prec_deriv + zz;

	    //retirar if adicionando o anel de zeros
	    if(inbounds(x_c, z_c, prec_deriv))
		Ps[zz][xx] = P[xzt(x_c, z_c, t-1)];
	    else
		Ps[zz][xx] = 0.0f;
	}
    }
    __syncthreads();
    

    float lap = deriv_x(Ps, x_s, z_s) + deriv_z(Ps, x_s, z_s);

    //assume que nao tem anel de zeros forcados por indice, entao usa as bordas como anel
    if(inbounds(x_g, z_g, prec_deriv))
	P[xzt(x_g, z_g, t)] = -P[xzt(x_g, z_g, t - 2)] + 2 * Ps[z_s][x_s] + cquad[xzt (x_g, z_g, 0)] * lap;
    else
	P[xzt(x_g,z_g,t)] = 0.0f;
}


__global__ void
somaFonte(float *P, int X, int Z, int T, int t, int *pos_source_x, int *pos_source_z, float *source, int n_source, int flip, int idx=-1)
{
    const int indexF = threadIdx.x + blockDim.x*threadIdx.y + blockIdx.x*(BLOCK_SIZE*BLOCK_SIZE);
    const int n = indexF; 

    if(n>=n_source || (idx!=-1 && idx!=n))
	return;

    float fonte;
    if(flip)
	fonte = source[n * T + T - 1 - t];
    else
	fonte = source[n*T + t];

    P[xzt(pos_source_x[n], pos_source_z[n], t)] += fonte;
}


__global__ void
gravaBufferSensores2ordem(float *P, float *recording, int X, int Z, int T, int t, int *pos_sensor_x, int *pos_sensor_z, int n_sensor)
{
    //coordenadas no bloco
    const int indexS = threadIdx.x + blockDim.x*threadIdx.y + blockIdx.x*(BLOCK_SIZE*BLOCK_SIZE);
    const int n = indexS; 
    if(n<n_sensor)
	recording[n*T + t] = P[xzt2(pos_sensor_x[n], pos_sensor_z[n], t)];
}


void
allocate_mem_simulate()
{
    cudaMalloc(&P, X * Z * 4 * sizeof (float));	//pressao direta
    cudaMalloc(&initial, X * Z * 2 * sizeof (float));	//pressao direta
    cudaMalloc(&cquad, X * Z * sizeof (float));	//campo de velocidades
    cudaMalloc(&source, T * n_source * sizeof (float));	//termos de fonte
    cudaMalloc(&record_buffer, T * n_sensor * sizeof (float));	//buffer dos sensores

    cudaMalloc(&pos_source_x, n_source*sizeof(int)); //posicoes das fontes
    cudaMalloc(&pos_source_z, n_source*sizeof(int)); //posicoes das fontes
    cudaMalloc(&pos_sensor_x, n_sensor*sizeof(int)); //posicoes dos sensores
    cudaMalloc(&pos_sensor_z, n_sensor*sizeof(int)); //posicoes dos sensores 

    cudaMallocHost(&recording_h, T*n_sensor*sizeof(float));
    //recording_h = (float*)malloc(T*n_sensor*sizeof(float));

    allocated = 1;
}


void
free_mem_simulate()
{
    cudaFree (P);
    cudaFree (initial);
    cudaFree (cquad);
    cudaFree (source);
    cudaFree (record_buffer);

    cudaFree(pos_source_x);
    cudaFree(pos_source_z);
    cudaFree(pos_sensor_x);
    cudaFree(pos_sensor_z);

    cudaFree(recording_h);

    allocated = 0;
}


extern "C" void
init_memory_sim(int x, int z, int t, float *cq, 
	int ns, int *ps_x, int *ps_z, int nm, int *pm_x, int *pm_z, 
	float *src, float *init, float **rec)
{
    X = x; 
    Z = z; 
    T = t; 
    n_source = ns; 
    n_sensor = nm;

    if(allocated)
	free_mem_simulate();

    cudaDeviceReset();

    allocate_mem_simulate();

    n_blocksX = X/BLOCK_SIZE;
    n_blocksZ = Z/BLOCK_SIZE;
    n_blocksS = n_sensor/(BLOCK_SIZE*BLOCK_SIZE);
    n_blocksF = n_source/(BLOCK_SIZE*BLOCK_SIZE);

    if(n_blocksX*BLOCK_SIZE != X)
	n_blocksX++;
    if(n_blocksZ*BLOCK_SIZE != Z)
	n_blocksZ++;
    if(n_blocksS*(BLOCK_SIZE*BLOCK_SIZE) != n_sensor) 
	n_blocksS++;
    if(n_blocksF*(BLOCK_SIZE*BLOCK_SIZE) != n_source) 
	n_blocksF++;


    blockGrid = {n_blocksX, n_blocksZ, 1};


    *rec = recording_h;
    cudaMalloc(&P, X * Z * 4 * sizeof (float));	//pressao direta
    cudaMemcpy(cquad, cq, X * Z * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(source, src, T * n_source * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_sensor_x, pm_x, n_sensor*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_sensor_z, pm_z, n_sensor*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_source_x, ps_x, n_source*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_source_z, ps_z, n_source*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(initial, init, X*Z*2*sizeof(float), cudaMemcpyHostToDevice);
}


extern "C" void
setCquad(float *cq)
{
    cudaMemcpy(cquad, cq, X * Z * sizeof (float), cudaMemcpyHostToDevice);
}


extern "C" void
set_source(int ns, int *sx, int *sz, float *src)
{
    n_source = ns;
    cudaMemcpy(pos_sensor_x, sx, ns * sizeof (int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_sensor_z, sz, ns * sizeof (int), cudaMemcpyHostToDevice);
    cudaMemcpy(source, src, ns * T * sizeof (float), cudaMemcpyHostToDevice);
}


extern "C" void
cuda_simulate2 (int en_out, int idx_source)
{
    FILE *pipeout;
    float *frame_buffer;
    if (en_out) 
    {
	char mpegCom[500];
	sprintf(mpegCom, "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt gray -s %ix%i -r 20 -i - -f mp4 -q:v 5 -an -vcodec h264 -crf 0 output/output1P.mp4 -nostats -loglevel quiet", X, Z);    
	pipeout = popen(mpegCom, "w");    
	cudaMallocHost(&frame_buffer, X*Z*sizeof(float));
    }

    //copia condicoes iniciais
    cudaMemset(P, 0, X*Z*4*sizeof(float));
    cudaMemcpy(P, initial, X*Z*2*sizeof(float), cudaMemcpyDeviceToDevice);

    for (int t = 0; t < T; t++)
    {
	//primeiros 2 frames sao condicao de contorno, logo nao calculados
	if (t > 1)
	{
	    simulateFrame <<<blockGrid, threadGrid>>> (P, cquad, X, Z, t);
	    cudaDeviceSynchronize ();

	    somaFonte<<<n_blocksF, threadGrid>>>(P, X, Z, T, t, pos_source_x, pos_source_z, source, n_source, 0, idx_source);
	    cudaDeviceSynchronize ();
	}
		
	// grava resultado nos sensores
	//gravaBufferSensores2ordem<<<1, n_sensor>>>(P, record_buffer, X, Z, T, t, pos_sensor_x, pos_sensor_z);
	gravaBufferSensores2ordem<<<n_blocksS, threadGrid>>>(P, record_buffer, X, Z, T, t, pos_sensor_x, pos_sensor_z, n_sensor);
	cudaDeviceSynchronize();

	if (en_out)
	{
	    cudaMemcpy(frame_buffer, &P[xzt(0,0,t)], X*Z*sizeof(float), cudaMemcpyDeviceToHost);
	    writeFramePipe (pipeout, frame_buffer, X, Z, t, pos_sensor_x, pos_sensor_z, n_sensor);
	}
    }

    if(en_out)
    {
	fflush(pipeout);
	pclose(pipeout);
	cudaFree(frame_buffer);
    }
    
    cudaMemcpy (recording_h, record_buffer, T * n_sensor * sizeof (float), cudaMemcpyDeviceToHost);
}

