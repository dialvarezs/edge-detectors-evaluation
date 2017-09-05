CC= g++
override CFLAGS+= -O2
OPENCV= `pkg-config --cflags --libs opencv`

all: canny clean

%.o: %.cpp
	$(CC) -c $< $(CFLAGS)

canny: CannyDetector_Demo.o
	$(CC) -o $@ $^ $(CFLAGS) $(OPENCV)

clean:
	rm -f *.o

remove:
	rm -f *.o canny
