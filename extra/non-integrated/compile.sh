#!/bin/bash
if [ $# -gt 0 ]; then
	if [ "$1" == "remove" ]; then
		make remove -C imgmat
		make remove -C edge_detector
		make remove -C performance
	fi
else
	make -C imgmat
	make -C edge_detector
	make -C performance
fi