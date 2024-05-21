#!/bin/bash
#SBATCH --partition=ashton
#SBACTH --qos=ashton
#SBATCH --job-name=mongarud-manual
#SBATCH --output=output_%j.log      # Standard output and error log
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

conda activate distmult
python3 run.py --embed_dim 750 --batch_size_train 300 --batch_size_test 150 --num_epochs 200 --lr 5e-4 --weight_decay 1e-3