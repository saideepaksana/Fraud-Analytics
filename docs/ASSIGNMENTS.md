# Assignment Overview

This page explains what each assignment is about, what files matter most, and what a newcomer should open first.

## Assignment 1: Graph Neural Networks

Assignment 1 studies fraud analytics as a graph learning problem. The data is represented as nodes and edges, where each node has features and a label. The goal is to predict node labels using graph neural network methods.

**Main question:** can graph structure improve classification compared with looking at each record alone?

**Important files:**

| File | Description |
|---|---|
| `Assignment1/GNN.ipynb` | Main notebook for graph neural network based node classification. |
| `Assignment1/graph_classification.ipynb` | Additional notebook comparing graph model variants. |
| `Assignment1/features.csv` | Node feature values. |
| `Assignment1/edges.csv` | Relationships between nodes. |
| `Assignment1/labels.csv` | Target labels for nodes. |
| `Assignment1/report.pdf` | Final submitted report. |

**Open first:** `Assignment1/GNN.ipynb`

## Assignment 2: Spectral Clustering

Assignment 2 focuses on unsupervised graph analysis. Instead of predicting known labels, it uses spectral clustering to discover groups of related nodes from the graph structure.

**Main question:** can the graph be divided into meaningful clusters using its connectivity pattern?

**Important files:**

| File | Description |
|---|---|
| `Assignment2/spectralCustering.ipynb` | Main spectral clustering notebook. |
| `Assignment2/spectral_graph_600nodes_edges.csv` | Input graph edge list. |
| `Assignment2/cluster_labels.csv` | Final cluster assignment for each node. |
| `Assignment2/report.pdf` | Final submitted report. |

**Open first:** `Assignment2/spectralCustering.ipynb`

## Assignment 3: Synthetic Transaction Data

Assignment 3 uses a Variational Autoencoder to learn patterns from transaction data and generate synthetic transactions. The synthetic data is then evaluated to check whether it preserves useful statistical and predictive properties.

**Main question:** can a generative model create realistic transaction data without directly copying the original records?

**Important files:**

| File | Description |
|---|---|
| `Assignment3/SyntheticDataGen.ipynb` | Main VAE training, generation, and evaluation notebook. |
| `Assignment3/customer_transactions_1500.csv` | Original transaction dataset. |
| `Assignment3/results/synthetic_transactions_vae_final.csv` | Generated synthetic transactions. |
| `Assignment3/results/evaluation_results.csv` | Evaluation results for synthetic data quality. |
| `Assignment3/results/training_history.json` | Training history saved from the VAE run. |
| `Assignment3/results/vae_model.pth` | Saved model checkpoint. |
| `Assignment3/report.pdf` | Final submitted report. |

**Open first:** `Assignment3/SyntheticDataGen.ipynb`

## Assignment 4: GAN-Based Fraud Detection

Assignment 4 treats fraud detection as an anomaly detection problem. Since fraudulent transactions are rare, GAN-based methods are used to learn the normal transaction pattern and identify unusual transactions as potential fraud.

**Main question:** can generative adversarial models help identify rare fraudulent transactions?

**Important files:**

| File | Description |
|---|---|
| `Assignment4/Team_36.ipynb` | Main final notebook. |
| `Assignment4/Team_36.pdf` | Final submitted report/export. |
| `Assignment4/KaggleRun.ipynb` | Kaggle-oriented experiment notebook. |
| `Assignment4/fraud final.ipynb` | Alternate final or draft notebook. |
| `Assignment4/fraud-assignment4-1.ipynb` | Earlier experiment notebook. |
| `Assignment4/fraud-assignment4-1-new.ipynb` | Updated experiment notebook. |
| `Assignment4/initial notebook.ipynb` | Initial experiment notebook. |
| `Assignment4/creditcard.csv` | Local dataset required to rerun the notebook. Ignored by Git. |

**Open first:** `Assignment4/Team_36.ipynb`

## How The Assignments Connect

The assignments move from graph-based fraud analysis to generative modeling:

1. Assignment 1 uses supervised graph learning.
2. Assignment 2 uses unsupervised graph clustering.
3. Assignment 3 generates synthetic transaction data.
4. Assignment 4 detects fraud as anomalies using GANs.

Together, they show different ways fraud analytics can be approached when the data is relational, imbalanced, sensitive, or difficult to label.
