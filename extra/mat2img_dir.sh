#!/bin/bash

DIRIN=$1
DIROUT=$2
NAME=`basename $1`

mkdir -p $DIROUT/$NAME
for i in `ls $DIRIN`
do
	imgmat/mat2img $DIRIN/$i $DIROUT/$NAME/`basename "${i%.*}"`.png
done
