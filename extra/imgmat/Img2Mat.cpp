/*
    Get grayscale matrix from a image
*/

#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    Mat image_bw;
    ofstream file_bw;

    if(argc<3)
    {
        cout << "./img2mat <imagen> <salida>(matriz)" << endl;
        return -1;
    }

    image_bw = imread(argv[1], 0);

    if(!image_bw.data)
        return -1;
    
    file_bw.open(argv[2]);

    file_bw << image_bw.rows << " " << image_bw.cols << "\n";

    file_bw << format(image_bw,Formatter::FMT_CSV); //cambiar Formatter::FMT_CSV por "csv" si es opencv2

    file_bw.close();

    return 0;
}