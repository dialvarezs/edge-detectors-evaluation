#!/bin/bash

if [ $# -lt 2 ]
	then
		echo "Usage: ./execute <image> <ground_truth> <output_dir>"
		exit
fi

IMG=$1
GT=$2
if [ $# -eq 2 ]
	then
		BASE_DIR="../"
	else
		BASE_DIR=$3
fi

NAME=`basename "${IMG%.*}"`
MATRIX_DIR="data/matrix/$NAME/"
EXEC_DIR="data/exec/$NAME""_`date +%Y%m%d-%H%M`/"

NOISE_INTS=10
NOISE_SMIN=0
NOISE_SMAX=0.2
NOISE_REPS=1

EDGE_MASK="../data/matrix/mask_sovel.dat"

rm -fr $BASE_DIR$MATRIX_DIR $BASE_DIR$EXEC_DIR
mkdir -p $BASE_DIR$MATRIX_DIR $BASE_DIR$EXEC_DIR
imgmat/img2mat $IMG $BASE_DIR$MATRIX_DIR$NAME.dat
sed -i '2,$ s/  *//g; s/,/ /g' $BASE_DIR$MATRIX_DIR$NAME.dat
sed '2,${s/[1-9][0-9]*/1/g}' $GT > $BASE_DIR$MATRIX_DIR$NAME""_edge.dat

START=$(date +%s.%N)
imgmat/noise_maker $BASE_DIR$MATRIX_DIR$NAME.dat $NOISE_INTS $NOISE_SMIN $NOISE_SMAX $NOISE_REPS
END=$(date +%s.%N)
echo -e "Noisy matrices creation: \t\t$(echo "scale=6; ($END - $START)/1" | bc) segs"


mkdir -p $BASE_DIR$MATRIX_DIR$NAME""_edges_g
FILE=$BASE_DIR$EXEC_DIR""execution_edge_g.dat
> $FILE

START=$(date +%s.%N)
for i in `ls $BASE_DIR$MATRIX_DIR$NAME""_noisy`
do
	edge_detector/edge_detector $BASE_DIR$MATRIX_DIR$NAME""_noisy/$i $BASE_DIR$MATRIX_DIR$NAME""_edges_g/${i%.*}_edge_g.dat g $EDGE_MASK >> $FILE
done
END=$(date +%s.%N)
echo -e "Edge detecttion (Gradient): \t\t$(echo "scale=6; ($END - $START)/1" | bc) segs"


mkdir -p $BASE_DIR$MATRIX_DIR$NAME""_edges_cv
FILE=$BASE_DIR$EXEC_DIR""execution_edge_cv.dat
> $FILE

START=$(date +%s.%N)
for i in `ls $BASE_DIR$MATRIX_DIR$NAME""_noisy`
do
	edge_detector/edge_detector $BASE_DIR$MATRIX_DIR$NAME""_noisy/$i $BASE_DIR$MATRIX_DIR$NAME""_edges_cv/${i%.*}_edge_cv.dat cv >> $FILE
done
END=$(date +%s.%N)
echo -e "Edge detecttion (CV): \t\t\t$(echo "scale=6; ($END - $START)/1" | bc) segs"


FILE=$BASE_DIR$EXEC_DIR""execution_performance_exh_edge_g.dat
> $FILE
START=$(date +%s.%N)
for i in `ls $BASE_DIR$MATRIX_DIR$NAME""_edges_g`
do
	performance/performance $BASE_DIR$MATRIX_DIR$NAME""_edges_g/$i $BASE_DIR$MATRIX_DIR$NAME""_edge.dat e >> $FILE
done
END=$(date +%s.%N)
echo -e "Performance (Exhaustive, Gradient): \t$(echo "scale=6; ($END - $START)/1" | bc) segs"


FILE=$BASE_DIR$EXEC_DIR""execution_performance_opt_edge_g.dat
> $FILE

START=$(date +%s.%N)
for i in `ls $BASE_DIR$MATRIX_DIR$NAME""_edges_g`
do
	performance/performance $BASE_DIR$MATRIX_DIR$NAME""_edges_g/$i $BASE_DIR$MATRIX_DIR$NAME""_edge.dat o >> $FILE
done
END=$(date +%s.%N)
echo -e "Performance (Optimized, Gradient): \t$(echo "scale=6; ($END - $START)/1" | bc) segs"


FILE=$BASE_DIR$EXEC_DIR""execution_performance_exh_edge_cv.dat
> $FILE

START=$(date +%s.%N)
for i in `ls $BASE_DIR$MATRIX_DIR$NAME""_edges_cv`
do
	performance/performance $BASE_DIR$MATRIX_DIR$NAME""_edges_cv/$i $BASE_DIR$MATRIX_DIR$NAME""_edge.dat e >> $FILE
done
END=$(date +%s.%N)
echo -e "Performance (Exhaustive, CV): \t\t$(echo "scale=6; ($END - $START)/1" | bc) segs"


FILE=$BASE_DIR$EXEC_DIR""execution_performance_opt_edge_cv.dat
> $FILE

START=$(date +%s.%N)
for i in `ls $BASE_DIR$MATRIX_DIR$NAME""_edges_cv`
do
	performance/performance $BASE_DIR$MATRIX_DIR$NAME""_edges_cv/$i $BASE_DIR$MATRIX_DIR$NAME""_edge.dat o >> $FILE
done
END=$(date +%s.%N)
echo -e "Performance (Optimized, CV): \t\t$(echo "scale=6; ($END - $START)/1" | bc) segs"
