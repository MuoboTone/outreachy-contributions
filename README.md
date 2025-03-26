# Predicting Mitochondrial Toxicity of Drugs Using Tox21’s SR-MMP Dataset

## Table of Contents

* [Introduction](#introduction)
* [Project Overview](#project-overview)
* [Getting Started](#getting-started)


## Introduction

**What are Mitochondria?**

A mitochondrion is a membrane-bound organelle found in the cytoplasm of almost all eukaryotic cells (cells with clearly defined nuclei), the primary function of which is to generate large quantities of energy in the form of adenosine triphosphate (ATP). Mitochondria are typically round to oval in shape and range in size from 0.5 to 10 μm. In addition to producing energy, mitochondria store calcium for cell signaling activities, generate heat, and mediate cell growth and death [1](https://www.britannica.com/science/mitochondrion)

**Mitochondrial Toxicity: Definition and Impact**

Mitochondrial Toxicity can be broadly defined as damage or dysfunction of the mitochondria, which can lead to various health problems, including muscle weakness, pancreatitis, and liver issues. 
Mitochondrial function is critical for health. This is demonstrated both by the large number of diseases caused by mutations in genes found in the nucleus and mitochondria, and by the critical role that mitochondrial dysfunction plays in a large number of chronic diseases [2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5681391/). 

Mitochondrial Dysfunction is linked to:
- Myopathy (muscle disease)
- Neurodegenerative Disorders (e.g Parkinson's disease)
- Cancer


## Project Overview:
This project builds a predictive machine learning model trained on the TOX 21's SR-MMP dataset, which contains qualitative toxicity measurements for 5,810 compounds. Given a drug SMILES string, the model predicts it's mitochondrial toxicity. 


**Dataset**
There are 12 toxic substances in Tox21, including the stress response effects (SR) and the nuclear receptor effects (NR). The SR includes five types (ARE, HSE, ATAD5, MMP, p53), and NR includes seven types (ER-LBD, ER, Aromatase, AhR, AR, AR-LBD, PPAR). Both the SR and NR effects are closely related to human health. For example, the activation of nuclear receptors can disrupt endocrine system function, and the activation of stress response pathways can lead to liver damage or cancer. The Tox21 database contains the results of high-throughput screening for these 12 toxic [effects](https://www.mdpi.com/1420-3049/24/18/3383). 

**Data Collection Method**
The data was collected using a multiplexed [two end points in one screen; MMP and adenosine triphosphate (ATP) content] quantitative high throughput screening (qHTS) approach combined with informatics tools to screen the Tox21 library of 10,000 compounds (~ 8,300 unique chemicals) at 15 concentrations each in triplicate to identify chemicals and structural features that are associated with changes in MMP in HepG2 cells. This allowed them generate a dataset to assess how chemicals reduce mitochondrial membrane potential [MMP](https://pmc.ncbi.nlm.nih.gov/articles/PMC4286281/). 

The Tox21 10K compound library, is a collaborative effort by several U.S. federal agencies (EPA, NIH, FDA, and others) to screen chemicals for toxicity-related biological activity. 

Assay specific threshold wasn't specified for SR-MMP (which is a subset of Tox21) but the 20% efficacy + p < 0.05 rule was the default for stress-response assays (e.g., p53, Nrf2/ARE). The SR-MMP assay used JC-1 dye(fluorescent dye), where a signal decrease = MMP loss. If a compound was classified as active (1) in the SR-MMP assay, it means the compound showed ≥20% signal reduction + p < 0.05 in replicate testing. Inactive compounds (0) did not meet these thresholds. 

## Getting Started

**Prerequisites**
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) or Anaconda installed on your system

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






