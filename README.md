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

2) Activate a conda environment corresponding to the experiment you wish to run.
```
(my_env) user@remotepc ~/nlp4neuro/data wget 
```

3) Run the experiment, or view experiment results, found in the "plot_results" folder.
```
(my_env) user@remotepc ~/nlp4neuro/code/experiment_1 bash run_experiment1.sh
```
```
(my_env) user@remotepc ~/nlp4neuro/code/plot_results python results1plot.py
```

## Conda environments

- Each experiment provides a .yml file and script for activating the corresponding environment.

## Assigning a GPU
- All jobs were run using the Duke Compute Cluster (DCC) in conjunction with SLURM workload manager.
- Please note that each experiment .sh script header data corresponds to this particular setup.
- You may alter each .sh script based on your work environment, or simply run the experiment .py file. 

```
(my_env) user@remotepc ~/nlp4neuro/code/experiment_1 python run_experiment1.py
```

## Web links for code/data
- NLP4Neuro code: [https://github.com/Naumann-Lab/nlp4neuro]
- NLP-4-Neuro data: [https://doi.org/10.7910/DVN/I8LULX]

Documentation written by Jacob Morra. For clarification you may contact me at [jacob.morra@duke.edu].
