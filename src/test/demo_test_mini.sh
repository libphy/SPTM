#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sptm
export PYTHONPATH=/opt/anaconda3/envs/sptm/lib/python3.7/site-packages
PREFIX=demo_test
nohup python run_eval.py --max-num-procs 4 --methods ours --doom-envs deepmind_small deepmind_large --params "$1" --exp-folder-prefix $PREFIX
bash plot_all.sh ../../experiments/${PREFIX}*/log.out

