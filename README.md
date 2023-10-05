# Code for generative prompting

## Requirements

Create a conda environment with all the dependencies using
```
conda create --name genprompt --file requirements.txt
```

Activate the conda environment: `conda activate genprompt`

## Reproducing results for SST2

This script runs the code for discriminative and generative prompting in all the settings for the sst2 dataset for all the models I experimented with. Look at run/sst2.sh to modify the code to run it only on `gpt-j-6b`.   
```
bash run/sst2.sh
```

This file will create various plots under `results/0923/`.




