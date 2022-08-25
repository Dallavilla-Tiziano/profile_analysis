#!/bin/bash
conda env create -f ./profiling_analysis.yml
conda activate pa_env
pip install supervenn
