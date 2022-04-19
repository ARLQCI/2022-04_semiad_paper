#! /bin/bash

## It looks for errors in the ODE propagation

## USAGE
## sh ./scripts/error_finder.sh

for f in ./data/benchmarks/*.log; do

    cat $f | grep Instability > /dev/null
    if [ $? -eq 0 ] ; then
	echo $f
	cat $f | grep Instability > /dev/null ; echo $?
    fi
done
