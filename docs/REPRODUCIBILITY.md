# Reproducibility Guide

This guide explains how to set up the project and rerun the coursework notebooks.

## 1. Create The Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter lab
```

## 2. Run Notebooks From Their Own Folder

The notebooks use local file paths. For example, Assignment 1 expects files like `features.csv`, `edges.csv`, and `labels.csv` to be nearby.

For the least friction, open each notebook from its assignment folder:

| Assignment | Notebook to run |
|---|---|
| Assignment 1 | `Assignment1/GNN.ipynb` |
| Assignment 2 | `Assignment2/spectralCustering.ipynb` |
| Assignment 3 | `Assignment3/SyntheticDataGen.ipynb` |
| Assignment 4 | `Assignment4/Team_36.ipynb` |

## 3. Add The Assignment 4 Dataset

Assignment 4 uses the Kaggle Credit Card Fraud Detection dataset. This file is large, so it is intentionally not tracked by Git.

Place the dataset here:

```text
Assignment4/creditcard.csv
```

Without this file, Assignment 4 will not run end to end.

## 4. Understand Generated Files

Some output files are already included because they are part of the coursework deliverables:

| File | Meaning |
|---|---|
| `Assignment2/cluster_labels.csv` | Cluster assignment output from Assignment 2. |
| `Assignment3/results/synthetic_transactions_vae_final.csv` | Synthetic transactions generated in Assignment 3. |
| `Assignment3/results/evaluation_results.csv` | Evaluation results for Assignment 3. |
| `Assignment3/results/training_history.json` | Training metrics from the VAE model. |
| `Assignment3/results/vae_model.pth` | Saved VAE model checkpoint. |

Rerunning notebooks may overwrite these files.

## 5. Expect Small Result Differences

Machine learning results can change slightly across runs because of random seeds, train/test splits, package versions, and CPU/GPU differences. The notebooks document the workflow, but exact numerical results may not always match perfectly after rerunning.
