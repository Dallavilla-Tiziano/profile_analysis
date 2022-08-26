#!/bin/bash
conda env create -f ./profiling_analysis.yml
eval "$(conda shell.bash hook)"
conda activate pa_env
pip install supervenn
