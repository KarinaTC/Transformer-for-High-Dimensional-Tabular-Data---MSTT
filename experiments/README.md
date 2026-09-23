# MSTT Experiments

This directory contains the scripts used to reproduce the experimental evaluation of the Multi-Module Transformer for Tabular Data (MSTT).

## Datasets

The experiments use datasets obtained from OpenML, as well as external datasets such as CUMIDA and LSVT.

To ensure a consistent evaluation protocol, the experiments use predefined cross-validation splits generated following the TabZilla workflow.

Rather than generating new data partitions during execution, the experimental scripts load the existing split indices to train, validate, and evaluate MSTT across the available folds.

## Dataset Structure

Each dataset must be stored in a separate directory containing the following files:

```text
datasets/
├── dataset_1/
│   ├── metadata.json
│   ├── split_indices.npy.gz
│   ├── X.npy.gz
│   └── y.npy.gz
│
├── dataset_2/
│   ├── metadata.json
│   ├── split_indices.npy.gz
│   ├── X.npy.gz
│   └── y.npy.gz
│
└── ...
```

The required files are described below:

| File | Description |
|---|---|
| `metadata.json` | Dataset metadata required by the experimental pipeline. |
| `split_indices.npy.gz` | Predefined cross-validation split indices. |
| `X.npy.gz` | Compressed NumPy array containing the input features. |
| `y.npy.gz` | Compressed NumPy array containing the target labels. |

## Preparing Additional Datasets

The experimental scripts are designed to work with datasets that follow the data format and cross-validation structure used by TabZilla.

If you wish to evaluate MSTT on an additional dataset that has not been previously processed, you must first prepare it using the TabZilla preprocessing pipeline.

This step generates the required dataset files and predefined cross-validation splits:

- `metadata.json`
- `split_indices.npy.gz`
- `X.npy.gz`
- `y.npy.gz`

Once these files have been generated, place them in a dedicated dataset directory following the structure described above.

This procedure also applies to external datasets that are not available through OpenML.

For further information on dataset preparation and split generation, please refer to the [TabZilla repository](https://github.com/naszilla/tabzilla).

## Cross-Validation Protocol

The experimental pipeline uses the predefined split indices stored in `split_indices.npy.gz`.

For each fold, the corresponding training, validation, and test sets are retrieved using these indices.

Using predefined splits ensures that the experimental evaluation can be reproduced without regenerating the data partitions and allows different models to be compared using the same data splits.

## Running the Experiments

Before running an experiment, ensure that the selected dataset follows the required directory structure.

The experimental scripts load the dataset features, target labels, metadata, and predefined split indices from the corresponding dataset directory.

MSTT is then trained and evaluated across the available cross-validation folds.
