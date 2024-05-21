#!/bin/bash
#SBATCH --partition=ashton
#SBACTH --qos=ashton
#SBATCH --job-name=mongarud-manual
#SBATCH --output=new-implementation/output_%j.log      # Standard output and error log
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

mkdir -p new-implementation

conda activate distmult
python3 run.py --embed_dim 750 --batch_size_train 300 --batch_size_test 150 --num_epochs 100 --lr 5e-4 --weight_decay 1e-3 --neg_sample_size 128