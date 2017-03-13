#ifndef _Matrix_Ops_H
#define _Matrix_Ops_H

int* load_matrix(const char* filename, int* width, int* height);
void save_matrix(const char* filename, int* matrix, int width, int height);

#endif