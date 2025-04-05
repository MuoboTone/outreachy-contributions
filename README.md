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


## Dataset Overview

There are 12 toxic substances in Tox21, including the stress response effects (SR) and the nuclear receptor effects (NR). The SR includes five types (ARE, HSE, ATAD5, MMP, p53), and NR includes seven types (ER-LBD, ER, Aromatase, AhR, AR, AR-LBD, PPAR). Both the SR and NR effects are closely related to human health. For example, the activation of nuclear receptors can disrupt endocrine system function, and the activation of stress response pathways can lead to liver damage or cancer. The Tox21 database contains the results of high-throughput screening for these 12 toxic [effects](https://www.mdpi.com/1420-3049/24/18/3383). 

### General Information
- Source: [Tox21 Data Challenge](https://tripod.nih.gov/tox21/challenge/data.jsp)
- Task: Binary classification (predicting toxicity)
- Target Variable: Toxicity label (1 = toxic, 0 = non-toxic)

### **Dataset Statistics**  

| Attribute                | Value                     |
|--------------------------|---------------------------|
| **Total Samples**        | 5,810                     |
| **Training Samples**     | 4,067 (~70%)                    |
| **Test Samples**         | 1,162 (~20%)                    |
| **Validation Samples**   | 581 (~10%)                       |
| **Positive Class (Toxic)** | 918 (~15.8%)                   |
| **Negative Class (Non-Toxic)** | 4,892  (~84.2%)            |

**Key Notes**  
- **Imbalanced dataset** (more non-toxic samples).  
- Pre-split into **70% train** / **20% test** / **10% validation**.   

### Dataset Structure
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| **Drug_ID** | `object`  | Unique row identifier (e.g., compound ID) |
| **Drug**    | `object`  | SMILES notation of the chemical compound |
| **Y**       | `float64` | Binary label: `0` (non-toxic) or `1` (toxic) |

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
   
```bash
git clone https://github.com/MuoboTone/outreachy-contributions.git
cd outreachy-contributions
```

2. Create the Conda Environment
```bash
conda env create -f environment.yml
```

3. Activate the Environment
```bash
conda activate myenv  # Check environment.yml for the name
```
4. Verify set-up
```bash
conda list
```

## Breakdown Of Implementation Process
1. **Data Acquisition**  
   - Downloaded the Tox21 SR-MMP dataset from the official source.  

2. **Exploratory Data Analysis (EDA)**  
   - Analyzed dataset structure, class imbalance, and missing values.    

3. **Featurization with Ersilia Models**  
   - Used [Ersilia](https://ersilia.io/) models to generate molecular embeddings/features.  
   - Processed SMILES strings into numerical representations for ML.  

4. **Model Building & Evaluation**  
   - Trained binary classification models (XGBoost).  
   - Addressed class imbalance via resampling/weighted loss.  
   - Evaluated performance using AUC-ROC, precision-recall, and F1-score.

### Environment Setup

```bash
conda create myenv python=3.9
conda activate myenv

# Install pyTDC for datasets
pip install pyTDC

# Install Ersilia Model Hub for featurization
pip install ersilia
```
#### 1. Data Acquisition

- Using [get_data.py](https://github.com/MuoboTone/outreachy-contributions/blob/main/scripts/get_data.py)

```python

from tdc.utils import retrieve_label_name_list 
label_list = retrieve_label_name_list('Tox21')

print(f"Available Assays are: {label_list}")
```
- Load the data from TDC and split it using TDC's default split method

```python
from tdc.single_pred import Tox

data = Tox(name='Tox21', label_name=label_list[10])
split = data.get_split() 

train_data = split['train']
valid_data = split['valid']
test_data = split['test']
```
- Download data into separate CSV files

```python
# Save all splits
split['train'].to_csv("tox21_train.csv", index=False)
split['valid'].to_csv("tox21_valid.csv", index=False)
split['test'].to_csv("tox21_test.csv", index=False)

# Save full unsplit dataset
full_data = data.get_data()
full_data.to_csv("tox21_full.csv", index=False)

#outputs are CSV files for each split and the entire dataset
```
#### 2. Exploratory Data Analysis (EDA)
- [initial_EDA.ipynb](https://github.com/MuoboTone/outreachy-contributions/blob/main/notebooks/exploratory%20analysis/initial_EDA.ipynb)
```python
df = pd.read_csv('data/tox21_full.csv')
df.info()
```
- Check for null values
```python
df = pd.read_csv("../data/tox21_test.csv")
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
print(df.isnull().sum())
plt.show()
```
```output
Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   Drug_ID  5810 non-null   object 
 1   Drug     5810 non-null   object 
 2   Y        5810 non-null   float64
```
- Classes split ratio
```output
Y
0.0    4892
1.0     918
Name: count, dtype: int64
```
**Split Method**
- Stratified.
- The stratified splitting method is important because an imbalance between negative/positive classes could lead to biased model evaluation if splits had different imbalance ratios.
- It also prevents a scenario where the minority class samples might be underrepresented in training. 

Here's the current ratio 

```output
Train class ratios:
 Y
0.0    3439
1.0     628

Test class ratios:
 Y
0.0    981
1.0    181

Validation class ratios:
 Y
0.0    472
1.0    109
```
- Check for data leakage between splits
```python
train = set(df_train['Drug'])  
val = set(df_valid['Drug'])
test = set(df_test['Drug'])

assert len(train & val) == 0, "Train/Val overlap detected!"
assert len(train & test) == 0, "Train/Test overlap detected!"
assert len(val & test) == 0, "Val/Test overlap detected!"
```
### Featurization with Ersilia Models

When selecting a featurizer for the Tox21 dataset, I considered several factors to ensure that the model can effectively capture the chemical information relevant to toxicity.
- Substructure Sensitivity: The featurizer should capture chemical substructures and functional groups that are known to influence toxicity.
- Computational Efficiency: While some deep learning-based representations can be powerful, they require more computational resources compared to traditional fingerprint methods.
- Preprocessing Requirements: I didn't want to choose a model that requires additional steps like normalization, scaling, or encoding categorical variables.
  
1. [Morgan Counts Fingerprints](https://github.com/ersilia-os/eos5axz):
   
   - Widely used, proven, and trusted in QSAR, virtual screening, and toxicity prediction tasks.
   - Circular substructure encoding around each atom helps the model recognize functional groups and reactive sites.
     
2. [DrugTax: Drug Taxonomy](https://github.com/ersilia-os/eos24ci):
   
   - Outputs features based on a molecule’s chemical taxonomy and composition, these are highly interpretable, which is a big plus when understanding why a compound might be toxic.
       
| Featurizer | Type | Number of Features | Input | Key Characteristics |
|------------|------|-------------------|-------|---------------------|
| DrugTax | Taxonomy classifier | 163 | SMILES | - Classifies molecules as organic/inorganic kingdom<br>- Includes detailed subclass information<br>- Counts chemical elements (carbons, nitrogens, etc.)<br>- Focuses on taxonomic classification of molecules          |
| Morgan Fingerprints (ECFP4) | Circular fingerprints | 2048 | SMILES | - Also known as extended connectivity fingerprints<br>- Circular search pattern from each atom    |

### Code Breakdown: [featurize.py](https://github.com/MuoboTone/outreachy-contributions/blob/main/scripts/featurize.py)**

- Step 1: Import Required Libraries
  ```python
  from ersilia import ErsiliaModel
  import os
  import pandas as pd
  ```
- Step 2: Define the Featurization Function
  ```python
  def featurize(model_ID, dataset_path, smiles_column):
  ```
The function takes three parameters:

- model_ID - the identifier for the specific Ersilia model to use
- dataset_path - path to the input dataset CSV file
- smiles_column - name of the column containing SMILES strings in the dataset

- Step 3: Model Setup and Data Loading
  ```python
  mdl = ErsiliaModel(model_ID)
  mdl.serve()
  #load data
  df = pd.read_csv(f"{dataset_path}") 
  # Extract just the Drug column containing SMILES
  smiles_df = df[[smiles_column]]
  # Save smiles to new CSV file
  smiles_df.to_csv("smiles_only.csv", index=False)
  ```

- Step 4: Prepare Output File
  ```python
  #create an empty CSV file to store featurized data
    pd.DataFrame().to_csv('featurized_data.csv', index=False)
    datasets = {
            "smiles_only.csv": "featurized_data.csv",
        }
  ```
  Sets up a dictionary mapping input files to output files (currently just one pair)

- Step 5: Run the Featurization
    ```python
    for input_file, output_file in datasets.items():
        # Check if the input file exists
        if os.path.exists(input_file): 

            # Run the model/featurization on the input file
            # Generating output to the specified output file
            mdl.run(input=input_file, output=output_file)
        else:
            # Raises an error if the input file is missing
            raise FileNotFoundError(f"Input file '{input_file}' not found!")
    ```

- Step 6: Clean Up Temporary Files
    ```python
    try:
        os.remove("smiles_only.csv")
    except FileNotFoundError:
        print(f"File not found")
        mdl.close()
    except PermissionError:
        print(f"Permission denied to delete file")
        mdl.close()
    except Exception as e:
        print(f"Error deleting file: {e}")
        mdl.close()
    ```

- Step 7: Close the Model and Confirm Completion
  ```python
  #close served model
  mdl.close() 
  print("Featurization Complete!")
  ```




