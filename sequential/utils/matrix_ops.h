#ifndef _Matrix_Ops_H
#define _Matrix_Ops_H

#define MASK_DIM 3

int* load_matrix(const char* filename, int* width, int* height);
int* load_mask(const char* filename);
void save_matrix(const char* filename, int* matrix, int width, int height);
void fill_borders(int* matrix, int width, int height);
int* noise_maker_saltpepper(int* matrix, int* noisy_matrix, int h, int w, float q);
int* noise_maker_multiplicative(int* matrix, int* noisy_matrix, int h, int w, float s);
int* noise_maker_additive(int* matrix, int* noisy_matrix, int h, int w, float s);
double gaussrand(float sigma, float mu);
int* mmalloc(int height, int width);
void mfree(int* matrix);
int median_qselect(int* values, int n);
void swap(int* a, int* b);

#endif