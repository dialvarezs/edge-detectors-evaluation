#!/bin/bash

if [ $# -lt 2 ]
    then
        echo "Usage: ./mresize.sh <image> <output_dir>"
        exit
fi

IMG=$1
NAME=`basename "${IMG%.*}"`
DIR=$2/$NAME

mkdir -p $DIR

for i in `seq 10 10 100`
do
    convert $IMG -resize `echo "sqrt($i)*10"|bc -l`% $DIR/$NAME""_`printf "%03d" $i`"".jpg
done
