#!/bin/bash
#SBATCH --partition=ashton
#SBACTH --qos=ashton
#SBATCH --job-name=mongarud-manual
#SBATCH --output=output_%j.log      # Standard output and error log
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

conda activate distmult
python3 run.py --embed_dim 512 --batch_size_train 300 --batch_size_test 150 --num_epochs 10 --lr 1e-3 --weight_decay 1e-7