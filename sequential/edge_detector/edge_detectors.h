#ifndef _Edge_Detectors_H
#define _Edge_Detectors_H

void edge_detector_g(int* img, int* edge, int width, int height, int* mask);
void edge_detector_cv(int* img, int* edge, int width, int height);
void edge_detector_mcv(int* img, int* edge, int width, int height);

void filter_med(int* img, int* img_filtered, int width, int height);
void filter_avg(int* img, int* img_filtered, int width, int height);

#endif