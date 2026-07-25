# Assignment 3: Synthetic Data Generation

This assignment generates synthetic transaction data using a Variational Autoencoder. The aim is to create new transaction-like records and evaluate whether they preserve useful patterns from the original data.

## Open First

Start with:

```text
SyntheticDataGen.ipynb
```

This notebook covers data preprocessing, VAE training, synthetic data generation, and evaluation.

## Files

| File | Description |
|---|---|
| `SyntheticDataGen.ipynb` | Main VAE workflow notebook. |
| `customer_transactions_1500.csv` | Original transaction dataset. |
| `results/synthetic_transactions_vae_final.csv` | Final synthetic transaction output. |
| `results/evaluation_results.csv` | Evaluation metrics for the generated data. |
| `results/training_history.json` | Saved training history. |
| `results/vae_model.pth` | Saved VAE model checkpoint. |
| `report.pdf` | Final assignment report. |
| `Assignment3.zip` | Submitted archive. |

## Notes

Run the notebook from this folder. Rerunning the notebook may overwrite files inside `results/`.
