#!/bin/bash
idx=0
N=36
res=20
while [ $idx -le $(($N-1)) ]
do
	python Serra09.py --range $res-$idx --wsub 5
	idx=$(($idx+1))
done
