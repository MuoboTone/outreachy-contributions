# Predicting Mitochondrial Toxicity of Drugs Using Tox21’s SR-MMP Dataset

## Table of Contents

* [Introduction](#introduction)
* [Project Overview](#project-overview)
* [Getting Started](#getting-started)
* [Breakdown Of Implementation Process](#breakdown-of-implementation-process)
* [Featurization with Ersilia Models](#featurization-with-ersilia-models)
* [Model Building and Evaluation](#model-building-and-evaluation)
* [Stretch Tasks](#stretch-task)
* [References](#references)


## Introduction

**What are Mitochondria?**

A mitochondrion is a membrane-bound organelle found in the cytoplasm of almost all eukaryotic cells (cells with clearly defined nuclei), the primary function of which is to generate large quantities of energy in the form of adenosine triphosphate (ATP). Mitochondria are typically round to oval in shape and range in size from 0.5 to 10 μm. In addition to producing energy, mitochondria store calcium for cell signaling activities, generate heat, and mediate cell growth and [death](https://www.britannica.com/science/mitochondrion).

**Mitochondrial Toxicity: Definition and Impact**

Mitochondrial Toxicity can be broadly defined as damage or dysfunction of the mitochondria, which can lead to various health problems, including muscle weakness, pancreatitis, and liver issues. 
Mitochondrial function is critical for health. This is demonstrated both by the large number of diseases caused by mutations in genes found in the nucleus and mitochondria, and by the critical role that mitochondrial dysfunction plays in a large number of chronic [diseases](https://pmc.ncbi.nlm.nih.gov/articles/PMC5681391/). 

Mitochondrial Dysfunction is linked to:
- Myopathy (muscle disease)
- Neurodegenerative Disorders (e.g Parkinson's disease)

## Project Overview:
This project develops a predictive machine learning model trained on the Tox21 SR-MMP dataset, which contains qualitative mitochondrial toxicity annotations for 5,810 chemical compounds. Given a compound's SMILES representation, the model predicts its potential mitochondrial toxicity. 

## Dataset Overview

There are 12 toxic substances in Tox21, including the stress response effects (SR) and the nuclear receptor effects (NR). The SR includes five types (ARE, HSE, ATAD5, MMP, p53), and NR includes seven types (ER-LBD, ER, Aromatase, AhR, AR, AR-LBD, PPAR). Both the SR and NR effects are closely related to human health. For example, the activation of nuclear receptors can disrupt endocrine system function, and the activation of stress response pathways can lead to liver damage or cancer. The Tox21 database contains the results of high-throughput screening for these 12 toxic [effects](https://www.mdpi.com/1420-3049/24/18/3383). 

### General Information
- Source: [Tox21 Data Task]([https://tripod.nih.gov/tox21/challenge/data.jsp](https://tdcommons.ai/single_pred_tasks/tox#tox21))
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
   - Addressed class imbalance via resampling/weighted class.  
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

- Using [get_data.py](https://github.com/MuoboTone/outreachy-contributions/blob/main/scripts/get_data.py) Script.

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
## Featurization with Ersilia Models

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

#### Code Breakdown: [featurize.py](https://github.com/MuoboTone/outreachy-contributions/blob/main/scripts/featurize.py)

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

### Featurization Output

#### Morgan Fingerprints Output Format
| Column Name  | Description |
|--------------|-------------|
| **Key**      | Unique identifier for each compound |
| **input**    | SMILES string representation of the compound |
| **dim_0000** | Count of substructure 1 in the molecule (integer ≥ 0) |
| **dim_0001** | Count of substructure 2 in the molecule (integer ≥ 0) |
| ...          | ... |
| **dim_2047** | Count of substructure 2048 |

*Note: Morgan Counts Fingerprint outputs **integer counts** for each substructure in the molecule, generated with radius 3.*

#### DrugTax Output Format
| Column Name | Description |
|-------------|-------------|
| **Key** | Unique identifier for each compound |
| **input** | SMILES string representation of the compound |
| **organic** | First taxonomic or molecular feature (binary 0/1) |
| **inorganic** | Second taxonomic or molecular feature (binary 0/1) |
| ... | ... |
| **char_/** | Last taxonomic or molecular feature (binary 0/1) |

*Note: DrugTax output includes 163 features representing taxonomy classification (organic/inorganic), subclasses, and counts of chemical elements (carbons, nitrogens, etc.).*
  
## Model Training and Evaluation

**ML Algorithm**
1. Morgan Fingerprint Model
     
     i. FLAML AutoML - estimators (LightGBM and RandomForest)
     
     ii. RandomForestClassifier
     
2. Drug Tax Model

     i. FLAML AutoML - estimator (LightGBM)

#### Code Breakdown: [Model Training](https://github.com/MuoboTone/outreachy-contributions/blob/main/notebooks/Model%20Training)

- Step 1: Import Required Libraries
  ```python
  from imblearn.over_sampling import SMOTE
  from sklearn.ensemble import RandomForestClassifier
  import pandas as pd
  from sklearn.metrics import classification_report
  import joblib
  from tdc import Evaluator
  ```
   
- Step 2: Load and prepare data.
   ```python
   # Get data
   train_data = pd.read_csv('../../data/MorganCount/tox21_train_featurized.csv').dropna()
   valid_data = pd.read_csv('../../data/MorganCount/tox21_valid_featurized.csv').dropna()
   test_data = pd.read_csv('../../data/MorganCount/tox21_test_featurized.csv').dropna()
   
   # Get X(features) and Y(target)
   X_train, y_train = train_data.filter(regex='^dim_.*'), train_data['Y']
   X_test, y_test = test_data.filter(regex='^dim_.*'), test_data['Y']
   X_valid, y_valid = valid_data.filter(regex='^dim_.*'), valid_data['Y']
   ```
    selected all columns starting with "dim_" from the featurized dataset.
      
- Step 3: Handle Class Imbalance with SMOTE
  ```python
  # Use SMOTE to oversample minority class
  smote = SMOTE(random_state=42)
  X_res, y_res = smote.fit_resample(X_train, y_train)
  ```
  SMOTE (Synthetic Minority Over-sampling Technique) is used to address class imbalance in datasets. It works by generating synthetic samples of the minority class rather than simply duplicating existing ones.

- Step 4: Initialize and Train RandomForest Model
  ```python
  # Train the model
  model = RandomForestClassifier(
    n_estimators=200,     # Number of trees
    max_depth=7,          # Maximum tree depth
    class_weight='balanced',  # Automatically adjusts for imbalanced classes
    random_state=42       # Reproducibility
   )
   
  model.fit(X_res, y_res, eval_set=[(X_valid, y_valid)], verbose=True)
  ```
  
- Step 5: Make Predictions and View Results
  ```python
  # Get prediction result
  y_test_pred = model.predict(X_test)
   
  # Display metrics
  print(classification_report(y_test, y_test_pred))
  ```
  
- Step 6: Get Evaluation Metrics
     ```python
     from typing import Dict, Any
     def evaluate_model(y_true, y_pred_proba, threshold: float = 0.5) -> Dict[str, float]:
          metrics = {
              'ROC-AUC': {'name': 'ROC-AUC', 'kwargs': {}},
              'PR-AUC': {'name': 'PR-AUC', 'kwargs': {}},
              'Accuracy': {'name': 'Accuracy', 'kwargs': {'threshold': threshold}},
              'Precision': {'name': 'Precision', 'kwargs': {'threshold': threshold}},
              'Recall': {'name': 'Recall', 'kwargs': {'threshold': threshold}},
              'F1': {'name': 'F1', 'kwargs': {'threshold': threshold}}
          }
          results = {}
          for metric_name, config in metrics.items():
              evaluator = Evaluator(name=config['name'])
              score = evaluator(y_true, y_pred_proba, **config['kwargs'])
              results[metric_name] = score
              print(f"{metric_name}: {score:.4f}")
          
          return results
   
      y_pred_proba = model.predict_proba(X_test)[:, 1]
      y_true = y_test
      
      evaluation_results = evaluate_model(y_true, y_pred_proba)
     ```

- Step 7. Save the Trained Model
  ```python
  model_filename = 'Morgan_trained_model.joblib'
  joblib.dump(model, model_filename)
  ```
  
### Results
### 🔍 Model Performance Comparison

| Metric            | Class | Morgan Model | DrugTax Model|
|-------------------|-------|---------|---------|
| **Precision**     | 0     | 0.90    | 0.90    |
|                   | 1     | 0.67    | 0.54    |
| **Recall**        | 0     | 0.96    | 0.93    |
|                   | 1     | 0.44    | 0.44    |
| **F1-Score**      | 0     | 0.93    | 0.91    |
|                   | 1     | 0.53    | 0.48    |
| **Accuracy**      | (0&1)  | 0.88    | 0.85    |
| **ROC-AUC**       |       | 0.82    | 0.89 |

**Observations**
- Both models had room for improvement. Although the Morgan Model has a higher ROC-AUC score(0.89-0.82).
- Precision and recall for both models is poor leading to a low overall F1 score (0.48(Drugtax) and 0.53 (Morgan))

These results led me use a more complex embedding to featurize my dataset. 

[Ersilia Compound Embeddings](https://github.com/ersilia-os/eos2gw4):
   
- The embeddings combine both physicochemical properties and bioactivity information. This fusion enables the model to capture complex relationships that might be missed by traditional descriptors alone.
- By incorporating Grover, Mordred, and ECFP descriptors during training, the featurizer captures both traditional substructural patterns and more nuanced chemical properties.
       
| Featurizer          | Type               | Number of Features | Input                     | Key Characteristics                          |
|---------------------|--------------------|--------------------|---------------------------|---------------------------------------------|
| Ersilia Embeddings  | Learned Embeddings | 1024   | SMILES strings | - Pre-trained deep learning representations<br>- Captures chemical structure and biological activity

#### Ersilia Embeddings Output Format
| Column Name | Description |
|-------------|-------------|
| **Key** | Unique identifier for each compound |
| **input** | SMILES string representation of the compound |
| **feature_0000** | First dimension of the embedding vector (continuous float) |
| **feature_0001** | Second dimension of the embedding vector (continuous float) |
| ... | ... |
| **feature_1023** | Last dimension of the embedding vector |

I also tested a different ML algorithm on the Morgan featurized data. Both Morgan and the Ersilia Compound Embeddings (ECE) model with the best performance across all metrics was trained using XGBoost with the follwing additional configurations: 

  | Parameter               |   ECE model| Morgan Model|
  |-------------------------|--------|--------|
  | `n_estimators`          | 300    | 200    |
  | `max_depth`             | 7      | 7   |
  | `learning_rate`         | 0.05   | 0.05    |
  | `gamma`                 | 0.5    | Default  |
  | `subsample`             | 0.8    | Default |
  | `colsample_bytree`      | 0.8    | Default|
  | `random_state`          | 42     | 42   |
  | `scale_pos_weight`      | 10     | 8   |
  | `early_stopping_rounds` | 10     | 10   |

  *Why These Values?*
      
  *n_estimators: Number of decision trees. More trees = stronger model but slower training.*
      
  *max_depth: Maximum depth of each tree. Deeper trees learn complex patterns but may overfit. 7 is deep enough to learn relationships but not too deep to overfit.*
      
  *learning_rate: How fast the model learns. Lower = more careful updates (better generalization but slower training). A value of 0.05 means small steps to avoid overshooting the best solution.*
      
  *early_stopping_rounds: Stops training if validation score doesn't improve for 10 rounds (prevents overfitting). Value 10 gives the model a few chances to improve before stopping.*

  *gamma: Minimum loss reduction required to split a node (regularization). Helps control overfitting.*
  
  *subsample: Fraction of training data randomly sampled for each tree. Using 80% of the data per tree introduces diversity, reducing overfitting*
  
  *colsample_bytree: Fraction of features randomly sampled for each tree.*
  
  *scale_pos_weight: Weight for the positive (minority) class in imbalanced datasets. A high value (10) strongly penalizes misclassifying the minority class.*

### 🔍 Model Performance Comparison (Morgan(XGBoost) vs ECE)
   | Metric       | Morgan Model | ECE Model  |
   |--------------|--------------|------------|
   | ROC-AUC      | 0.8859       | **0.8897** |
   | PR-AUC       | 0.6369       | **0.6754** |
   | Accuracy     | 0.8509       | **0.8718** |
   | Precision    | 0.5161       | **0.5684** |
   | Recall       | 0.7072       | **0.7348** |
   | F1 Score     | 0.5967       | **0.6410** |

Their performance can be closely evaluated using the confusion matrix plots:
![ECE cong](https://github.com/user-attachments/assets/118a7720-20fa-4ca7-a45b-997b6ac76a1a)
![output mogan conf](https://github.com/user-attachments/assets/d20aca65-ea80-40eb-9a50-721af85d36e0)

#### 📝 Key Observations
ECE Model:
- True Negatives (TN): 869 (correctly identified negatives)
- False Positives (FP): 112 (incorrectly labeled as positive)
- False Negatives (FN): 46 (incorrectly labeled as negative)
- True Positives (TP): 135 (correctly identified positives)

Morgan Model:
- True Negatives (TN): 859 (correctly identified negatives)
= False Positives (FP): 120 (incorrectly labeled as positive)
= False Negatives (FN): 53 (incorrectly labeled as negative)
- True Positives (TP): 128 (correctly identified positives)

#### 💡Overall Assessment:
   The ECE model outperforms the Morgan model across all four confusion matrix metrics. It has better detection of both positive and negative classes, with fewer errors in both directions.
   This suggests that the ECE approach might be capturing more meaningful patterns in the data for this classification task compared to the Morgan approach.

   The models performs well, with an of ROC-AUC (0.88+) matching many published Tox21 challenge models, though teir PR-AUC (~0.63 - 0.67) shows room for improvement. While the current results are promising, 
   it can be better thus they aren't satisfactory.

*Both models are more thoroughly evaluated in the [evaluation notebooks](https://github.com/MuoboTone/outreachy-contributions/tree/main/notebooks/model%20evaluation).* 

## Stretch Task
In order to further evaluate both models, I downloaded an external dataset published on [American Chemical Society](https://acs.figshare.com/articles/dataset/Evaluation_of_in_Vitro_Mitochondrial_Toxicity_Assays_and_Physicochemical_Properties_for_Prediction_of_Organ_Toxicity_Using_228_Pharmaceutical_Drugs/7492784?file=13881770). The article studies whether mitochondrial toxicity screening assays can effectively predict drug-induced organ damage in humans, particularly for the liver, heart, and kidney. The dataset contained 228 compounds, they studied (73 hepatotoxicants, 46 cardiotoxicants, 49 nephrotoxicants, and 60 non-toxic compounds) along with their measured mitochondrial toxicity results from both screening assays (isolated rat liver mitochondria and glucose-galactose grown HepG2 cells).

I selected 112 compounds from the dataset with 50 positive(toxic) and 62 negative(non-toxic). I tried to eliminate class imbalance and get sufficient examples of both toxic and non-toxic compounds. I ensured non of the compounds were present in my tox_21 sr-mmp dataset. So these are all compounds my model hasn't seen before.

I evaluated both the Ersilia Compound Embeddings (ECE) model and the Morgan Fingerprint model on the external dataset.

#### Results

| Metric       | Morgan Model | ECE Model  | 
|--------------|--------------|------------|
| ROC-AUC      | 0.8332       | **0.8574** |
| PR-AUC       | 0.7329       | **0.7691** |
| Accuracy     | 0.7411       | 0.7411     |
| Precision    | **0.8000**   | 0.7838     | 
| Recall       | 0.5600       | **0.5800** |
| F1 Score     | 0.6588       | **0.6667** |

#### Model Performance Comparison: Test Data vs External Validation

| Metric       | Morgan (Test) | Morgan (External) | ECE (Test) | ECE (External) |
|--------------|---------------|-------------------|------------|----------------|
| ROC-AUC      | 0.8859        | 0.8332 (-5.95%)   | 0.8897     | 0.8574 (-3.63%) |
| PR-AUC       | 0.6369        | 0.7329 (+15.07%)  | 0.6754     | 0.7691 (+13.87%)|
| Accuracy     | 0.8509        | 0.7411 (-12.90%)  | 0.8718     | 0.7411 (-15.00%)|
| Precision    | 0.5161        | 0.8000 (+55.01%)  | 0.5684     | 0.7838 (+37.90%)|
| Recall       | 0.7072        | 0.5600 (-20.81%)  | 0.7348     | 0.5800 (-21.07%)|
| F1           | 0.5967        | 0.6588 (+10.41%)  | 0.6410     | 0.6667 (+4.00%) |

#### 📊 Confusion matrices: 
![ECE external conf](https://github.com/user-attachments/assets/e6415189-467e-46af-a0a2-7400be51b5b8)
![morgan external conf](https://github.com/user-attachments/assets/8311f112-4830-41e8-8f6e-1ca4c140868b)

#### 📝 Key Observations
- Generalization Gap: Both models perform worse on external data (lower ROC-AUC/Accuracy), but ECE degrades less.
- Precision ↑ + Recall ↓: Models are overly conservative in external data (prioritizing fewer false positives at the cost of missing true positives).
- models are biased toward predicting "non-toxic" compounds due to the class imbalance in the training data.
- PR-AUC Improvement: Suggests better precision-recall balance in external data (maybe positives are "easier" to identify).

### 📚 Recommendations
1. Address Recall Drop: Use threshold tuning to balance precision/recall (e.g., lower decision threshold to increase recall).
2. Cost-Benefit Analysis:
   - If false positives are costly (e.g., toxic compounds mislabeled as safe), current precision is acceptable. 
   - If false negatives are worse (e.g., missing toxic compounds), improve recall at the expense of precision.






### References

- [Attene-Ramos, M. S., Huang, R., Michael, S., Witt, K. L., Richard, A., Tice, R. R., Simeonov, A., Austin, C. P., & Xia, M. (2015). Profiling of the Tox21 chemical collection for mitochondrial function to identify compounds that acutely decrease mitochondrial membrane potential. Environmental Health Perspectives, 123(1), 49-56.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4286281/)
- [Yuan, Q.; Wei, Z.; Guan, X.; Jiang, M.; Wang, S.; Zhang, S.; Li, Z. Toxicity Prediction Method Based on Multi-Channel Convolutional Neural Network. Molecules 2019, 24, 3383.](https://doi.org/10.3390/molecules24183383)
- [Rana, Payal; Aleo, Michael D.; Gosink, Mark; Will, Yvonne (2018). Evaluation of in Vitro Mitochondrial Toxicity Assays and Physicochemical Properties for Prediction of Organ Toxicity Using 228 Pharmaceutical Drugs. ACS Publications.](https://doi.org/10.1021/acs.chemrestox.8b00246.s001https://acs.figshare.com/articles/dataset/Evaluation_of_in_Vitro_Mitochondrial_Toxicity_Assays_and_Physicochemical_Properties_for_Prediction_of_Organ_Toxicity_Using_228_Pharmaceutical_Drugs/7492784?file=13881770)
- [Meyer JN, Chan SSL. Sources, mechanisms, and consequences of chemical-induced mitochondrial toxicity. Toxicology. 2017 Nov 1;391:2-4. doi: 10.1016/j.tox.2017.06.002. Epub 2017 Jun 13. PMID: 28627407; PMCID: PMC5681391.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5681391/)
  




