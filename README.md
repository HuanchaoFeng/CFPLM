# CFPLM: Improve protein-RNA interaction prediction with a collaborative framework powered by language models

## Main Model Location

The primary model implementation for this paper is located in:
`/benchmark/model.py` or
`/lncRNA/model.py` or
`/mirna/model.py`
they are all the same model.

## Environment Setup

To run CFPLM, the following environment configuration is required. A detailed `environment.yml` file can be used for setup:

- **torch == 2.3.0**
- **python == 3.11.4**
- **GPU**: NVIDIA RTX A6000
- **CUDA Version**: 12.7

## Dataset Structure

The repository includes three main directories, each containing datasets specific to different RNA types and benchmark scenarios:

- `benchmark/`: Contains benchmark datasets, including sequence data and interaction data.
- `lncRNA/`: Contains datasets related to long non-coding RNA (lncRNA) and its interactions with proteins.
- `mirna/`: Contains datasets related to microRNA (miRNA) and its interactions with proteins.

*Note: All datasets in these directories have been preprocessed into the required feature vectors for direct use with the model.*

## Usage

### 1. Running the Model

To start the prediction process, execute the `launch.py` file located in each of the dataset directories:

```bash
# For benchmark datasets
python benchmark/launch.py

# For lncRNA datasets
python lncRNA/launch.py

# For miRNA datasets
python mirna/launch.py
```

### 2. Extracting Custom Features

If you need to extract protein or RNA features from your own sequence data (instead of using the preprocessed feature vectors), run the following scripts:

- For RNA feature extraction:
  ```bash
  python rna_feature.py
  ```

- For protein feature extraction:
  ```bash
  python protein_feature.py
  ```

## Notes

- Ensure that the required GPU and CUDA version are available to achieve optimal performance.
- For further details on the model architecture or dataset preprocessing, refer to the associated research paper.
