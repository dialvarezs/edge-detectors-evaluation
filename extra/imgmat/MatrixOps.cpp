#include <fstream>

using namespace std;

int* load_matrix(const char* filename, int* width, int* height)
{
	ifstream file;
	int* matrix;
	int h,w;

	file.open(filename);

    file >> h >> w;

    matrix = new int[h*w];

    for(int i=0; i<h; i++)
    	for(int j=0; j<w; j++)
    		file >> matrix[i*w+j];

    file.close();

    *width = w;
	*height = h;
    return matrix;
}

void save_matrix(const char* filename, int* matrix, int width, int height)
{
	ofstream file;

	file.open(filename);

	file << height << " " << width << endl;

    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            file << matrix[i*width+j];
            if(j<width-1)
                file << " ";
        }
        file << "\n";
    }

    file.close();
}