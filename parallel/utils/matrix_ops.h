#ifndef _Matrix_Ops_H
#define _Matrix_Ops_H

#define MASK_DIM 3

int* load_matrix(const char* filename, int* width, int* height);
int* load_mask(const char* filename);
void save_matrix(const char* filename, int* matrix, int width, int height);
void fill_borders(int* matrix, int width, int height);
void noise_maker_multiplicative(int* matrix, int* noisy_matrix, int h, int w, float s);
double gaussrand(float sigma, float mu);
int* mmalloc(int height, int width);
void mfree(int* matrix);

#endif