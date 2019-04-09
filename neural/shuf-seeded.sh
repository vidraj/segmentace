#!/bin/bash

set -e

if [ "$#" -lt 1 ] || [ '-' = "${1:0:1}" ]; then
        echo -e "Permute lines of a file or STDIN based on a string seed.\nUsage:\n\t$0 SEED [shuf params...]\n\nTo prevent stupid mistakes, SEED starting with '-' is\n interpreted as a misplaced option and rejected.\n\nSee also: man shuf" >&2
        exit 1
else
	seed="$1"
	shift
	shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null) "$@"
fi
