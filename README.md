# Predicting Mitochondrial Toxicity of Drugs Using Tox21’s SR-MMP Dataset

## Table of Contents

* [Introduction](#introduction)
* [Project Overview](#project-overview)
* [Getting Started](#getting-started)
* [Breakdown Of Implementation Process](#breakdown-of-implementation-process)


## Introduction

**What are Mitochondria?**

A mitochondrion is a membrane-bound organelle found in the cytoplasm of almost all eukaryotic cells (cells with clearly defined nuclei), the primary function of which is to generate large quantities of energy in the form of adenosine triphosphate (ATP). Mitochondria are typically round to oval in shape and range in size from 0.5 to 10 μm. In addition to producing energy, mitochondria store calcium for cell signaling activities, generate heat, and mediate cell growth and death [1](https://www.britannica.com/science/mitochondrion).

**Mitochondrial Toxicity: Definition and Impact**

Mitochondrial Toxicity can be broadly defined as damage or dysfunction of the mitochondria, which can lead to various health problems, including muscle weakness, pancreatitis, and liver issues. 
Mitochondrial function is critical for health. This is demonstrated both by the large number of diseases caused by mutations in genes found in the nucleus and mitochondria, and by the critical role that mitochondrial dysfunction plays in a large number of chronic diseases [2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5681391/). 

Mitochondrial Dysfunction is linked to:
- Myopathy (muscle disease)
- Neurodegenerative Disorders (e.g Parkinson's disease)
- Cancer


## Project Overview:
This project builds a predictive machine learning model trained on the TOX 21's SR-MMP dataset, which contains qualitative toxicity measurements for 5,810 compounds. Given a drug SMILES string, the model predicts it's mitochondrial toxicity. 


## Dataset

There are 12 toxic substances in Tox21, including the stress response effects (SR) and the nuclear receptor effects (NR). The SR includes five types (ARE, HSE, ATAD5, MMP, p53), and NR includes seven types (ER-LBD, ER, Aromatase, AhR, AR, AR-LBD, PPAR). Both the SR and NR effects are closely related to human health. For example, the activation of nuclear receptors can disrupt endocrine system function, and the activation of stress response pathways can lead to liver damage or cancer. The Tox21 database contains the results of high-throughput screening for these 12 toxic [effects](https://www.mdpi.com/1420-3049/24/18/3383). 

### Data Collection Method

The data was collected using a multiplexed [two end points in one screen; MMP and adenosine triphosphate (ATP) content] quantitative high throughput screening (qHTS) approach combined with informatics tools to screen the Tox21 library of 10,000 compounds (~ 8,300 unique chemicals) at 15 concentrations each in triplicate to identify chemicals and structural features that are associated with changes in MMP in HepG2 cells. This allowed them generate a dataset to assess how chemicals reduce mitochondrial membrane potential [MMP](https://pmc.ncbi.nlm.nih.gov/articles/PMC4286281/). 

The Tox21 10K compound library, is a collaborative effort by several U.S. federal agencies (EPA, NIH, FDA, and others) to screen chemicals for toxicity-related biological activity. 

Assay specific threshold wasn't specified for SR-MMP (which is a subset of Tox21) but the 20% efficacy + p < 0.05 rule was the default for stress-response assays (e.g., p53, Nrf2/ARE). The SR-MMP assay used JC-1 dye(fluorescent dye), where a signal decrease = MMP loss. If a compound was classified as active (1) in the SR-MMP assay, it means the compound showed ≥20% signal reduction + p < 0.05 in replicate testing. Inactive compounds (0) did not meet these thresholds. 

### Tabular Summary

| **Feature**          | **Description** |
|----------------------|----------------|
| **Assay**            | SR-MMP (subset of Tox21) |
| **Threshold Rule**   | 20% efficacy + *p* < 0.05 (default for stress-response assays like p53, Nrf2/ARE) |
| **Detection Method** | JC-1 fluorescent dye (signal decrease = mitochondrial membrane potential (MMP) loss) |
| **Active (1)**       | ≥20% signal reduction + *p* < 0.05 in replicate testing |
| **Inactive (0)**     | Did not meet the above thresholds |


## Getting Started

**Prerequisites**
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) or Anaconda installed on your system
- [Docker](https://docs.docker.com/engine/install/)
- Git Large File Storage (LFS) `conda install git-lfs -c conda-forge`
- GitLFS Activated `git-lfs install`

**Installation Guide**

1. Clone the Repository
```
git clone https://github.com/MuoboTone/outreachy-contributions.git
cd outreachy-contributions
```

2. Create the Conda Environment
```
conda env create -f environment.yml
```

3. Activate the Environment
```
conda activate myenv  # Check environment.yml for the name
```


**Additional Notes**
- this guide will be updated as the project goes on.

## Breakdown Of Implementation Process

This section includes the process of setting up my conda environment, downloading the Tox21 SR-MMP dataset, and featurizing it using DrugTax.

### Environment Setup

```
conda create myenv python=3.9
conda activate myenv

# Install pyTDC for datasets
pip install pyTDC

# Install Ersilia Model Hub for featurization
pip install ersilia

# I fetched the DrugTax model from ersilia hub
ersilia fetch eos24ci
```
### Downloading Tox21 SR-MMP Dataset

- Using [get_data_notebook.ipynb](https://github.com/MuoboTone/outreachy-contributions/blob/main/notebooks/get_data_notebook.ipynb)

```
#first I printed the label list to confirm the name of my assay of interest
from tdc.utils import retrieve_label_name_list 
label_list = retrieve_label_name_list('Tox21')
print(f"Available Assays are: {label_list}")
```
- Next I loaded the data from TDC and split it using TDC's default split method

```
from tdc.single_pred import Tox

data = Tox(name='Tox21', label_name=label_list[10])
split = data.get_split() 

train_data = split['train']
valid_data = split['valid']
test_data = split['test']
```
- After loading the datasets I downloaded it into seperate csv files

```
# Save all splits
split['train'].to_csv("tox21_train.csv", index=False)
split['valid'].to_csv("tox21_valid.csv", index=False)
split['test'].to_csv("tox21_test.csv", index=False)

# Save full unsplit dataset
full_data = data.get_data()
full_data.to_csv("tox21_full.csv", index=False)

#outputs are csv files for each split and the entire dataset
```
- Next i got some information about the data

```
import pandas as pd

df = pd.read_csv('data/tox21_full.csv')
df.info()

#output is info about the size of the data and all columns
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5810 entries, 0 to 5809
Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   Drug_ID  5810 non-null   object 
 1   Drug     5810 non-null   object 
 2   Y        5810 non-null   float64
dtypes: float64(1), object(2)
memory usage: 136.3+ KB
```

### Justification for Split Choice:

- After carrying out minimal exploratory analysis on the dataset, I realized there was class imbalance in the target variable (Y column).

```
Y
0.0    4892
1.0     918
Name: count, dtype: int64
```
- I discovered that TDC's random split method was stratified.
- The stratified splitting method is important because imbalance between negative/positive classes could lead to biased model evaluation if splits had different imbalance ratios.
- It also prevents a scenario where the minority class samples might be underrepresented in training. 

Here's the current ratio 
```
Train class ratios:
 Y
0.0    3439
1.0     628
Name: count, dtype: int64
Test class ratios:
 Y
0.0    981
1.0    181
Name: count, dtype: int64
Validation class ratios:
 Y
0.0    472
1.0    109
Name: count, dtype: int64
```
### Featurization

### Justification for DrugTax: Drug taxonomy:
I initially tried to use the Ersilia compound embeddings model but the 1024 features output were a struggle to manage. It also kept returning null values at some point. When this happened I decided to search for a simpler model with smaller outputs but with just as detailed features. 

- It takes small molecule representations in SMILES format as input and allows the simultaneously extraction of taxonomy information and key features.
- DrugTax can identify molecules belonging to 31 superclasses, including organic molecules like organoheterocyclic, organosulphur, lipid, allene, benzenoid, phenylpropanoid, organic acid, alkaloid, organic salt, etc., and inorganic molecules like organohalogen, organometallic, organic nitrogen, nucleotide, etc.
- It's 163 feature outputs for each SMILE input made it easy for me to handle.


## Using DrugTax (eos24ci) from ersilia model hub

```
#NOTE: This script runs successfully I already installed Ersilia Model Hub and fetched the model using the CLI

# import main class
from ersilia import ErsiliaModel
import os
```
- Next I instantiated the model and served it

```
mdl = ErsiliaModel("model_ID") #use your model id
mdl.serve()
```
- Extracted the "Drug" column from my dataset and made it its own csv file to serve as input

```
import pandas as pd

#load your data
path = "" #replace with dataset file name without the csv
df = pd.read_csv(f"{path}.csv") 

# Extract just the Drug column containing SMILES
smiles_df = df[["SMILES_column"]] #replace with name of column holding smiles

# Save to new CSV file
smiles_df.to_csv("smiles_only.csv", index=False) #creates a new csv file containing only the smile strings
```
- Created a dictionary to map input file to it's corresponding output file

```
#defines a dictionary that maps input files to their corresponding output files for featurization

pd.DataFrame().to_csv('smiles_featurized.csv', index=False) #create an empty CSV file

datasets = {
        "smiles_only.csv": "smiles_featurized.csv",
    }
```
- Iterated through the datasets dictionary and ran the model/featurization on the input file

```
for input_file, output_file in datasets.items():
    if os.path.exists(input_file): # Check if the input file exists

        # Run the model/featurization on the input file
        # Generating output to the specified output file
        mdl.run(input=input_file, output=output_file)
    else:
        raise FileNotFoundError(f"Input file '{input_file}' not found!")  # Raise an error if the input file is missing
```

- Next I merged the featurized data to my original dataset using the "Drug" and "input" columns 
```
import pandas as pd
original_data = pd.read_csv(f"{path}.csv")
features = pd.read_csv("smiles_featurized.csv")

# Merge data on column  containing SMILES
combined_data = pd.merge(
    original_data,
    features,
    left_on="Drug",    # Column in original dataset
    right_on="input",  # Column in DrugTax output
    how="left"         # Keeps all rows from original_data
)


 
combined_data.to_csv(f"{path}_featurized.csv", index=False)  #create an empty CSV file and adds combined data
```
- Finally I closed the model 

```
mdl.close() #close served model
```




