#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include "regression.h"

void imshow(float *img, int X, int Y)
{

    FILE *output;

    output = fopen(".tmpimage.tmp", "wb+");
    float max = img[0], min = img[0];
    for(int i = 0;i<X*Y; i++)
    {
	float value = img[i];
	if(value>max) max = value;
	if(value<min) min = value;
    }
    float offset = -min;
    float scale = 255.0f/(max-min);
    unsigned char *buffer;
    buffer = (unsigned char*)malloc(sizeof(unsigned char*)*X*Y);
    for(int i = 0;i<X*Y; i++)
    {
	buffer[i] = (img[i] + offset)*scale;
    }
    fwrite(buffer, sizeof(unsigned char), X*Y, output);
    free(buffer);
    fclose(output);

    char command[200];
    sprintf(command, "cat .tmpimage.tmp | convert -size %dx%d -depth 8 -format GRAY GRAY:- .tmpimage.png", X, Y);
    int saida = 0;
    saida += system(command);
    saida += system("eog .tmpimage.png");
    saida += system("rm .tmpimage.tmp .tmpimage.png");
    if(saida)
    	printf("Erro escrevendo arquivos de imagem!");
    	
    return;
}


void writeFrame(float *img, int X, int Y, int t, int* sensor_x, int* sensor_z, int n_sensor)
{
    FILE *output;
    char nome[50];
    float *frame_h;

    frame_h = (float*)malloc(sizeof(float)*X*Y);
    memcpy(frame_h, img, sizeof(float)*X*Y);

    sprintf(nome, "output/images/saida%05d.blob", t);
    output = fopen(nome, "wb+");
    float max = img[0], min = img[0];
    for(int i = 0;i<X*Y; i++)
    {
	float value = img[i];
	if(value>max) max = value;
	if(value<min) min = value;
    }
    float offset = -min;
    float scale = 255.0f/(max-min);
    unsigned char *buffer;
    buffer = (unsigned char*)malloc(sizeof(unsigned char*)*X*Y);
    for(int i = 0;i<X*Y; i++)
    {
	buffer[i] = (img[i] + offset)*scale;
    }
    for(int i = 0;i<n_sensor; i++)
    {
	buffer[sensor_x[i] + X*sensor_z[i]] = 255;
    }
    fwrite(buffer, sizeof(unsigned char), X*Y, output);
    free(buffer);
    free(frame_h);
    fclose(output);

    return;
}


void writeFramePipe(FILE* pipe, float *img, int X, int Y, int t, int* sensor_x, int* sensor_z, int n_sensor)
{
    int *sensor_z_h, *sensor_x_h;

    sensor_x_h = (int*)malloc(sizeof(int)*n_sensor);
    sensor_z_h = (int*)malloc(sizeof(int)*n_sensor);

    cudaMemcpy(sensor_x_h, sensor_x, sizeof(int)*n_sensor, cudaMemcpyDeviceToHost);
    cudaMemcpy(sensor_z_h, sensor_z, sizeof(int)*n_sensor, cudaMemcpyDeviceToHost);

    // float max = 9, min = -9;  // OK bounds for second order acoustic equation
    float max = 2, min = -2;  // OK bounds for first order acoustic equation system w/o d_x, d_z
    // float max = [0], min = img[0];
    // for(int i = 0;i<X*Y; i++)
    // {
	// float value = img[i];
	// if(value>max) max = value;
	// if(value<min) min = value;
    // }
    //printf("%f, %f;\n", min, max);
    float offset = -min;
    float scale = 255.0f/(max-min);
    unsigned char *buffer;
    buffer = (unsigned char*)malloc(sizeof(unsigned char*)*X*Y);
    for(int i = 0;i<X*Y; i++)
    {
	buffer[i] = (img[i] + offset)*scale;
    if (buffer[i] < 0) // clipping for clarity
    {
        buffer[i] = 0;
    }
    else if (buffer[i] > 255)
    {
        buffer[i] = 255;
    }
    }
    for(int i = 0;i<n_sensor; i++)
    {
	buffer[sensor_x_h[i] + X*sensor_z_h[i]] = 255;
    }
    fwrite(buffer, sizeof(unsigned char), X*Y, pipe);
    free(buffer);
    free(sensor_x_h);
    free(sensor_z_h);

    return;
}
