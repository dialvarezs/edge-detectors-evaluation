/*
	Save matrix image to grayscale image
*/

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "MatrixOps.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{	
	Mat image;
	int* matrix;
	int h, w, mult = 1;

	if(argc < 3)
	{
		cout << "./img2mat <matriz> <salida>(imagen) [<mult>]" << endl;
		return -1;
	}

	if(argc == 4)
		mult = atoi(argv[3]);

	matrix = load_matrix(argv[1], &w, &h);

	image = Mat(h, w, CV_32SC1, matrix);

	image *= mult; //makes the edges more visible

	imwrite(argv[2], image);

	return 0;
}