<img src="https://github.com/user-attachments/assets/5ba45c6e-999b-43a0-881c-689adb8b99d7" width="100" height="50">

# NLP4Neuro
Off-the-shelf, pre-trained large language models applied to neural population decoding.

## Getting started

- Models can be fine-tuned with or without a GPU. For testing, we used an Nvidia RTX A6000 card.
- You will need to install Python 3.9, including package managers i.e. Anaconda and pip.

## Conda environments

- Each experiment provides a .yml file and script for activating the corresponding environment. All packages which are necessary for a given experiment will be installed.
- To run each code, you will need to point the directories (e.g. results directory, data directory) to the corresponding "data" folder.

## Data availability
- All data is available at https://doi.org/10.7910/DVN/EFP1IL. Please download all data and unzip the folder into the "data" folder.

## Running each experiment
- To run an experiment, depending on your working environment, run the corresponding shell script or simply type
```
python run_experimentX.py
```

For assistance or clarification, please create an issue or contact me by mail at jacob.morra@duke.edu.
