<img src="https://github.com/user-attachments/assets/5ba45c6e-999b-43a0-881c-689adb8b99d7" width="450">

# NLP4Neuro
Off-the-shelf, pre-trained large language models applied to neural population decoding.

## Getting started

1) Download the data from here [https://doi.org/10.7910/DVN/I8LULX] into the "data" directory.

From a GNU terminal window:
```
wget -r --no-parent data "https://dataverse.harvard.edu/api/datasets/:persistentId/dirindex?persistentId=doi:10.7910/DVN/I8LULX&folder=exp1-4_data"
```

From Windows Powershell:
```
Invoke-WebRequest -Uri "https://dataverse.harvard.edu/api/datasets/:persistentId/dirindex?persistentId=doi:10.7910/DVN/I8LULX&folder=exp1-4_data" -OutFile "exp1-4_data"
```

Alternatively, there is a sample download_dataverse powershell script provided in the "data" folder. 


2) Update the config.yaml file with the locations of your data and results folders.


3) Activate a conda environment corresponding to the experiment you wish to run.
```
user@remotepc ~/nlp4neuro/code/experiment_1 bash create_jjm_env_nlp1.sh
```

4) Run the experiment, or view experiment results, found in the "plot_results" folder.
```
(jjm_env_nlp1) user@remotepc ~/nlp4neuro/code/experiment_1 bash run_experiment1.sh
```
```
(jjm_env_nlp3) user@remotepc ~/nlp4neuro/code/plot_results python exp3_create_pcaplots.py
```

## Conda environments

- Each experiment provides a .yml file and script for activating the corresponding environment.

## Assigning a GPU
- All jobs were run using the Duke Compute Cluster (DCC) in conjunction with SLURM workload manager.
- E.g. in run_experiment_1.sh, we ran jobs requiring 64 GB of RAM, 8 CPUs per task, and an A6000 GPU

```
#!/bin/bash
#SBATCH --job-name=nlp4neuro          # Job name
#SBATCH --output=logs/nlp4neuro_exp1_%j.out # Standard output file (%j will be replaced by the job ID)
#SBATCH --error=logs/nlp4neuro_exp1_%j.err  # Standard error file
#SBATCH --time=72:00:00                    # Max run time (adjust as needed)
#SBATCH --mem=64G                         # Memory request per node
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL               # Email notifications for job done & fail
#SBATCH --mail-user=jacob.morra@duke.edu # Your email for notifications
#SBATCH -p scavenger-gpu --gres=gpu:6000_ada:1 #Select the GPU
```

- Please note that each experiment .sh script header data corresponds to this particular setup.
- You may alter each .sh script based on your work environment, or simply run the experiment .py file. 

```
(my_env) user@remotepc ~/nlp4neuro/code/experiment_1 python run_experiment1.py
```

## Web links for code/data
- NLP4Neuro code: [https://github.com/Naumann-Lab/nlp4neuro]
- NLP-4-Neuro data: [https://doi.org/10.7910/DVN/I8LULX]


## Systems/envs used for testing
1) Lenovo Legion Slim 5 14APH8, 32 GB RAM, NVidia RTX 4060 8 GB, Windows 11 Home
2) Duke Compute Cluster (DCC) remote access, NVidia A6000 via Windows Powershell

Documentation written by Jacob Morra. For clarification you may contact me at [jacob.morra@duke.edu].
