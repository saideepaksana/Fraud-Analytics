# Fraud Analytics

This repository contains my coursework for a **Fraud Analytics** course. The work is organized as a sequence of four assignments, each exploring a different way to model fraud-related data using graphs, clustering, synthetic data generation, and anomaly detection.

Most of the project is written as Jupyter notebooks. Each assignment folder contains the notebook work, the data used for that assignment, generated outputs, and the final report or submission files.

## What This Repository Covers

| Assignment | Main idea | What it does |
|---|---|---|
| `Assignment1/` | Graph neural networks | Uses graph structure, node features, and labels to perform node classification. |
| `Assignment2/` | Spectral clustering | Finds communities or clusters in a graph using spectral clustering. |
| `Assignment3/` | Synthetic data generation | Trains a Variational Autoencoder (VAE) to generate synthetic transaction data. |
| `Assignment4/` | Fraud anomaly detection | Uses GAN-based methods such as AnoGAN and BiGAN for credit card fraud detection. |

## Start Here

If you are new to this repo, read the files in this order:

1. [`docs/ASSIGNMENTS.md`](docs/ASSIGNMENTS.md) - simple explanation of each assignment.
2. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) - how to set up the environment and rerun notebooks.
3. The `README.md` inside each assignment folder - quick guide to that assignment's files.
4. The notebooks and PDFs - full implementation and report details.

## Folder Structure

```text
.
├── Assignment1/          # Graph neural network based node classification
├── Assignment2/          # Spectral clustering on graph data
├── Assignment3/          # Synthetic transaction generation using VAE
├── Assignment4/          # GAN-based fraud detection
├── docs/                 # Project documentation
├── requirements.txt      # Python dependencies
└── README.md             # Repository overview
```

## Main Notebooks

| Assignment | Recommended notebook |
|---|---|
| Assignment 1 | `Assignment1/GNN.ipynb` |
| Assignment 2 | `Assignment2/spectralCustering.ipynb` |
| Assignment 3 | `Assignment3/SyntheticDataGen.ipynb` |
| Assignment 4 | `Assignment4/Team_36.ipynb` |

## Running The Notebooks

Create a Python environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then start Jupyter:

```bash
jupyter lab
```

Open the assignment folder you want to review and run the notebook from that folder. This helps the notebook find its local CSV files correctly.

## Data Note

Most small datasets are included in the repository. The large Kaggle credit card dataset for Assignment 4 is not committed.

Expected local path:

```text
Assignment4/creditcard.csv
```

Download the dataset from Kaggle and place it at that path if you want to rerun Assignment 4.

## Notes For Reviewers

- The notebooks are the main source of implementation detail.
- The PDF files are the submitted report/export versions.
- Zip files are preserved as submission archives.
- Some Assignment 4 notebooks are draft or experiment versions; `Team_36.ipynb` is the recommended file to review first.
