CC= g++-5
override CFLAGS+= -O2 -Wall -std=c++11
DEPS= MatrixOps.h
OBJ= MatrixOps.o
OPENCV= `pkg-config --cflags --libs opencv`

all: mat2img img2mat noise_maker clean

%.o: %.cpp $(DEPS)
	$(CC) -c $< $(CFLAGS)

mat2img: Mat2Img.o $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(OPENCV)

img2mat: Img2Mat.o
	$(CC) -o $@ $^ $(CFLAGS) $(OPENCV)

noise_maker: NoiseMaker.o $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o

remove:
	rm -f *.o mat2img img2mat noise_maker
