#ifndef __REGRESSION_H__
#define __REGRESSION_H__

#include<stdio.h>
#include<stdlib.h>


void imshow(float *img, int X, int Y);
void writeFrame(float *img, int X, int Y, int t, int* sensor_x, int* sensor_z, int n_sensor);
void writeFramePipe(FILE* pipe, float *img, int X, int Y, int t, int* sensor_x, int* sensor_z, int n_sensor);

extern "C" void hilbert(float *signal, float *signal90, float *envelope, int N);

#endif
