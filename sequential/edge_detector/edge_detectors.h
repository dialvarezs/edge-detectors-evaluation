#ifndef _Edge_Detectors_H
#define _Edge_Detectors_H

int* edge_detector_g(int* img, int* edge, int width, int height, int* mask);
int* edge_detector_cv(int* img, int* edge, int width, int height);
int* edge_detector_mcv(int* img, int* edge, int width, int height);

int* filter_med(int* img, int* img_filtered, int width, int height);
int* filter_avg(int* img, int* img_filtered, int width, int height);

#endif