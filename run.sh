#!/bin/bash

#SBATCH --job-name=para
#SBATCH --output=res.txt
#SBATCH --error=error.txt
#SBATCH --time=120:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../../miniconda3/etc/profile.d/conda.sh
conda activate qwen

python ./submission.py 