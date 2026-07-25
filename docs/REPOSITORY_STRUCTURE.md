# Repository Structure

The repository is organized by assignment. This keeps each notebook close to its data, outputs, report, and submission archive.

```text
.
├── Assignment1/
│   ├── GNN.ipynb
│   ├── graph_classification.ipynb
│   ├── features.csv
│   ├── edges.csv
│   ├── labels.csv
│   └── report.pdf
├── Assignment2/
│   ├── spectralCustering.ipynb
│   ├── spectral_graph_600nodes_edges.csv
│   ├── cluster_labels.csv
│   └── report.pdf
├── Assignment3/
│   ├── SyntheticDataGen.ipynb
│   ├── customer_transactions_1500.csv
│   ├── results/
│   └── report.pdf
├── Assignment4/
│   ├── Team_36.ipynb
│   ├── Team_36.pdf
│   └── experiment notebooks
├── docs/
├── requirements.txt
└── README.md
```

## What Belongs Where

| Type of file | Location |
|---|---|
| Main notebooks | Inside the matching `Assignment*/` folder. |
| Assignment datasets | Inside the matching `Assignment*/` folder. |
| Generated outputs | Inside the assignment folder, or inside a local `results/` folder. |
| Reports | Inside the matching assignment folder. |
| General documentation | Inside `docs/`. |
| Environment dependencies | `requirements.txt` at the repository root. |

## Canonical Review Files

Use these as the first files to open:

| Assignment | Open first |
|---|---|
| Assignment 1 | `Assignment1/GNN.ipynb` |
| Assignment 2 | `Assignment2/spectralCustering.ipynb` |
| Assignment 3 | `Assignment3/SyntheticDataGen.ipynb` |
| Assignment 4 | `Assignment4/Team_36.ipynb` |

## Archive And Draft Files

The zip files are preserved as submission snapshots. Draft notebooks are kept where they provide useful experiment history, especially in Assignment 4.

For a clean review, start with the canonical files listed above before opening drafts or archives.
