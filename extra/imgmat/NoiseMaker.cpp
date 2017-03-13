/*
    Noise generator
*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <tuple>
#include <sys/stat.h>
#include "MatrixOps.h"

using namespace std;

int* noise_maker(int* matrix, int* noisy_matrix, int h, int w, float s);
pair<string, string> split_filename(const string& str);
string fixed_length(const int i, const int length);
void usage();

int main(int argc, char** argv )
{	
    int* matrix;
    int* noisy_matrix;
    int h, w, n, m;
    float sigma_min, sigma_max, sigma, step;
    string dir, name;

	if(argc<6)
	{
        usage();
        return -1;
	}

    tie(dir, name) = split_filename(string(argv[1]));
    name.erase(name.size() - 4); //assumes filename end with ".dat" or some extension with lenght 3

    n = atoi(argv[2]);
    sigma_min = atof(argv[3]);
    sigma_max = atof(argv[4]);
    m = atoi(argv[5]);
    step = (sigma_max - sigma_min)/n;

    mkdir((dir+"/"+name+"_noisy").c_str(), 0755);
    dir += "/"+name+"_noisy";

    matrix = load_matrix(argv[1], &w, &h);
    noisy_matrix = new int[h*w];
    sigma = sigma_min;

    for(int i=0; i<n; i++)
    {
        for(int j=1; j<=m; j++)
        {
            noisy_matrix = noise_maker(matrix, noisy_matrix, h, w, sigma);

            string filename = dir+"/"+name+"_s0"+to_string(sigma).substr(2,3)+"_"+fixed_length(j,3)+".dat";

            save_matrix(filename.c_str(), noisy_matrix, w, h);
        }
        sigma += step;
    }

    return 0;
}

int* noise_maker(int* matrix, int* noisy_matrix, int h, int w, float s)
{
    random_device rd;
    mt19937 gen(rd());

    normal_distribution<> d(1,s);

    for(int i=0; i<h; i++)
        for(int j=0; j<w; j++)
        {
            noisy_matrix[i*w+j] = matrix[i*w+j]*(round(1000*d(gen))/1000);
            if(noisy_matrix[i*w+j] > 255)
                noisy_matrix[i*w+j] = 255;
        }

    return noisy_matrix;
}

pair<string, string> split_filename(const string& str)
{
    string dir, name;
    size_t split = str.find_last_of("/");
    dir = str.substr(0, split);
    name = str.substr(split+1);

    return make_pair(dir, name);
}

string fixed_length(const int i, const int length)
{
    ostringstream ostr;

    if (i < 0)
        ostr << '-';
    ostr << setfill('0') << setw(length) << (i < 0 ? -i : i);
    
    return ostr.str();
}

void usage()
{
    cout << "./noise_maker <matriz> <intervalos> <sigma_min> <sigma_max> <repeticiones>" << endl;
}