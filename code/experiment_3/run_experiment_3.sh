#!/bin/bash

#SBATCH --job-name=nlp4neuro          # Job name
#SBATCH --output=logs/nlp4neuro_exp_3_%j.out # Standard output file (%j will be replaced by the job ID)
#SBATCH --error=logs/nlp4neuro_exp_3_%j.err  # Standard error file
#SBATCH --time=72:00:00                    # Max run time (adjust as needed)
#SBATCH --mem=64G                         # Memory request per node
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL               # Email notifications for job done & fail
#SBATCH --mail-user=jacob.morra@duke.edu # Your email for notifications
#SBATCH -p scavenger-gpu --gres=gpu:6000_ada:1 #Select the GPU


# previous gpu used was gpu:a5000:1, now 6000_ada:1
# Go to right folder
# cd /hpc/home/jjm132/nlp4neuro/code

# Activate the venv
source /hpc/group/naumannlab/jjm132/miniconda3/bin/activate jjmenv

# Run your python script
python experiment_3.py
