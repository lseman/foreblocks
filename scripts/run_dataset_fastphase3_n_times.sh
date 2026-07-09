#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$ROOT/estudo_rebeca"
RUNS=""
SOURCE="hidro"
SEQ="50"
HORIZON="5"

usage() {
	echo "Usage: $(basename "$0") N [-s SOURCE] [-seq SEQ] [-horizon HORIZON]"
	echo "Example: $(basename "$0") 10 -s hidro -seq 50 -horizon 5"
}

while (($# > 0)); do
	case "$1" in
	-s | -source | --source)
		if (($# < 2)); then
			usage
			exit 1
		fi
		SOURCE="$2"
		shift 2
		;;
	-seq | --seq)
		if (($# < 2)); then
			usage
			exit 1
		fi
		SEQ="$2"
		shift 2
		;;
	-horizon | --horizon)
		if (($# < 2)); then
			usage
			exit 1
		fi
		HORIZON="$2"
		shift 2
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		if [[ -z "$RUNS" ]]; then
			RUNS="$1"
			shift
		else
			echo "Unknown argument: $1"
			usage
			exit 1
		fi
		;;
	esac
done

COMMAND=(python3 dataset_run.py -s "$SOURCE" -seq "$SEQ" -horizon "$HORIZON" -fastphase3)

if [[ -z "$RUNS" ]] || ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
	usage
	exit 1
fi

if ! [[ "$SEQ" =~ ^[0-9]+$ ]] || [[ "$SEQ" -lt 1 ]]; then
	echo "-seq must be a positive integer"
	exit 1
fi

if ! [[ "$HORIZON" =~ ^[0-9]+$ ]] || [[ "$HORIZON" -lt 1 ]]; then
	echo "-horizon must be a positive integer"
	exit 1
fi

if [[ ! -f "$WORKDIR/dataset_run.py" ]]; then
	echo "dataset_run.py not found in $WORKDIR"
	exit 1
fi

cd "$WORKDIR"

for ((run = 1; run <= RUNS; run++)); do
	echo "[$run/$RUNS] Running: ${COMMAND[*]}"
	"${COMMAND[@]}"
done
