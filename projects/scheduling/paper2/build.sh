#!/usr/bin/env bash
set -euo pipefail
export TEXMFVAR=/tmp/foreblocks-paper2-texmf
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build-pdflatex.log
bibtex main > build-bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build-pdflatex.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build-pdflatex.log
