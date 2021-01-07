#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sptm
export PYTHONPATH=/opt/anaconda3/envs/sptm/lib/python3.7/site-packages
PREFIX=demo_test
nohup python run_eval.py --max-num-procs 4 --methods ours --doom-envs deepmind_small deepmind_large branching open_space_two open_space_five star_maze office1 columns office2 topological_star_easier --params "$1" --exp-folder-prefix $PREFIX
bash plot_all.sh ../../experiments/${PREFIX}*/log.out

