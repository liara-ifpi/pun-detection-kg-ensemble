# Knowledge Graph-Enhanced Ensemble for Portuguese Pun Detection

This repository implements a complete experimental pipeline for Portuguese pun detection, combining structured knowledge graph representations with traditional textual features through a soft-voting ensemble model.

The project integrates two main components:

- Knowledge graph construction
- Graph-enhanced ensemble training

---

## Repository Structure

- `kg_builds.ipynb` — Knowledge graph construction pipeline
- `kg_ensemble_training.ipynb` — Graph-enhanced ensemble training pipeline
- `corpus/` — Labeled dataset (train, validation, test)
- `outputs/` — Generated knowledge graphs and related artifacts

---

## Part I — Knowledge Graph Construction

The graph construction pipeline:

1. Loads the labeled corpus.
2. Normalizes and filters tokens.
3. Builds three types of knowledge graphs:
   - Co-occurrence Graph (Count-based)
   - PPMI Graph (Positive Pointwise Mutual Information)
   - Pun-Context Graph (from positive examples only)
4. Exports graphs in reusable formats.
5. Generates statistical summaries and optional visualizations.

### Graph Variants

**Co-occurrence Graph**
- Window-based token co-occurrence
- Edge weights = frequency counts

**PPMI Graph**
- Edge weights computed using Positive Pointwise Mutual Information

**Pun-Context Graph**
- Connects pun-labeled tokens with contextual tokens

### Graph Outputs

- `.gpickle` serialized NetworkX graphs
- CSV edge lists
- Statistical summaries
- Optional static and interactive visualizations

---

## Part II — Graph-Enhanced Ensemble Training

The training pipeline:

1. Loads train/validation/test splits.
2. Vectorizes texts using TF-IDF (unigrams + bigrams).
3. Extracts structural features from generated knowledge graphs.
4. Normalizes and scales graph-based features.
5. Fuses textual and structural features.
6. Trains a Soft Voting Ensemble classifier.
7. Evaluates performance using accuracy, precision, recall and F1-score.

---

## Model Architecture

### Textual Representation

- TF-IDF vectorization
- N-gram range: (1, 2)
- Portuguese stopword filtering

### Graph-Based Features

Extracted structural features include:

- Number of nodes
- Number of edges
- Average degree
- Graph density
- Average edge weight

These features encode relational properties derived from lexical interactions.

---

## Ensemble Strategy

The final classifier is a Soft Voting Ensemble composed of:

- Random Forest
- Logistic Regression
- Linear SVM (probability-enabled)

Hyperparameters may be optimized via randomized search.

---

## Experimental Configurations

Supported fusion scenarios:

- TF-IDF + Co-occurrence Graph
- TF-IDF + PPMI Graph
- TF-IDF + Pun-Context Graph
- TF-IDF + All Graphs Combined

Each configuration is evaluated independently.

---

## Requirements

Developed and tested with:

- Python 3.10+
- Google Colab (recommended) or local Jupyter environment

Main libraries:

- pandas
- numpy
- networkx
- scikit-learn
- scipy
- tqdm
- matplotlib
- pyvis
- ipywidgets

Dependencies are installed directly within the notebooks via `pip`.

---

## Reproducibility

To reproduce the experiments:

1. Run `kg_builds.ipynb` to generate the knowledge graphs.
2. Run `kg_ensemble_training.ipynb` to train and evaluate the model.

---

## Outputs

The complete pipeline generates:

- Knowledge graph artifacts
- Trained ensemble models
- Classification reports
- Evaluation tables

These results support ablation studies and graph-feature comparisons.

---

## License

This project is released under the MIT License.  
See the `LICENSE` file for details.

---

<div align=center>
  <h3>Authors</h3>
  <a href="https://github.com/avelando">
    <img src="https://img.shields.io/badge/avelando-GitHub-black">
  </a>
  <a href="https://github.com/camiwr">
    <img src="https://img.shields.io/badge/camiwr-GitHub-white">
  </a>
  <a href="https://github.com/CarlHenry670">
    <img src="https://img.shields.io/badge/CarlHenry670-GitHub-black">
  </a>
  <a href="https://github.com/rafaelanchieta">
    <img src="https://img.shields.io/badge/rafaelanchieta-GitHub-white">
  </a>

  <h6>Copyright (c) 2026 Laboratório de Inteligência Artificial, Robótica e Automação</h6>
</div>
