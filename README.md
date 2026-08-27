# TACO 🌮

![Python](https://img.shields.io/badge/python-3.9--3.12-blue)
[![PyPI](https://img.shields.io/pypi/v/tabpfn-taco)](https://pypi.org/project/tabpfn-taco/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/OpenReview-84mfkGDxYh-8c1b13?logo=openreview&style=flat-square)](https://openreview.net/pdf?id=84mfkGDxYh)

Official repository for the paper
**"End-to-End Compression for Tabular Foundation Models"**.

Tabular foundation models such as TabPFN learn in context, taking the training
data as input at inference time. Because their attention mechanism scales
quadratically with dataset size, training and inference get expensive and the
models struggle on large tables — and the common workarounds, subsampling rows
or capping table size, give up accuracy. TACO instead learns to compress the
training set in a latent space, shrinking the context the model has to attend
over. We show that this gives up to 94x faster inference and up to
97% lower memory use than the underlying tabular transformer, with no significant
loss in predictive performance.

## Quick Start

### Prerequisites

- Python 3.9-3.12

### Installation

The distribution is named `tabpfn-taco`, while its Python import package is
named `taco`.

Install the latest release from PyPI:

```bash
pip install tabpfn-taco
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/machinelearningnuremberg/TACO.git
cd TACO
pip install -e .
```

Alternatively, use [uv](https://docs.astral.sh/uv/), which installs the project
in editable mode using the checked-in lockfile:

```bash
uv sync
```

This example shows how to evaluate **TabPFN-TACO** with compression and
**TabPFN-POT** without compression using `TACOClassifier` on the scikit-learn
Breast Cancer dataset.

### Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from taco.model.tabpfn_arch.taco_classifier import TACOClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y,
)

# TabPFN-TACO with compression
clf_taco = TACOClassifier(
    use_compressor=True,
    row_compression_percentage=4,
    fit_mode="fit_preprocessors",
)

clf_taco.fit(X_train, y_train)

prediction_probabilities = clf_taco.predict_proba(X_test)
print("TabPFN-TACO ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

predictions = prediction_probabilities.argmax(axis=1)
print("TabPFN-TACO Accuracy:", accuracy_score(y_test, predictions))

# TabPFN-POT without compression
clf_pot = TACOClassifier(
    use_compressor=False,
    fit_mode="fit_preprocessors",
)

clf_pot.fit(X_train, y_train)

prediction_probabilities = clf_pot.predict_proba(X_test)
print("TabPFN-POT ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

predictions = prediction_probabilities.argmax(axis=1)
print("TabPFN-POT Accuracy:", accuracy_score(y_test, predictions))
```

### Large Chunked Inference

For large datasets, use `fit_with_chunking`. See
[`examples/taco_chunking.py`](examples/taco_chunking.py) for a runnable example:

```bash
uv run python examples/taco_chunking.py
```

### Pretraining

To pretrain TabPFN-TACO and TabPFN-POT from scratch, use the training
configurations provided in:

- `scripts/train_stage1_taco_random.sh`
- `scripts/train_stage1_pot.sh`

Install the training extras before running these scripts:

```bash
pip install -e ".[train]"
```

Or with uv:

```bash
uv sync --extra train
```

### Checkpoints

The released **TabPFN-TACO** and **TabPFN-POT** weights are downloaded
automatically from Hugging Face the first time you use `TACOClassifier`, so no
manual download or `checkpoint_path` is required. The weights are published at
<https://huggingface.co/zabergjg/TabPFN-TACO>.

## License and Attribution

TACO-original code is released under the BSD 3-Clause License (see [LICENSE](LICENSE)).
This repository also includes code derived from TabPFN and TabICL, which remain
under their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
the bundled texts in [`LICENSES/`](LICENSES).

Public checkpoint and model artifacts are released under the names **TabPFN-TACO**
and **TabPFN-POT**. The released checkpoints are trained from scratch and do not
redistribute or use TabPFN or TabICL pretrained weights.

> **Built with PriorLabs-TabPFN**


## Acknowledgments

TACO builds on the open-source work of two projects, and we thank their authors:

- **TabPFN** ([Prior Labs](https://github.com/PriorLabs/TabPFN)) — the tabular
  foundation model architecture that TACO compresses.
- **TabICL** ([Soda team @ Inria](https://github.com/soda-inria/tabicl)) — whose
  prior-generation and pretraining code TACO's training pipeline builds on.

## Citation

Author contribution: Guri Zabërgja and Rafiq Kamel contributed equally to the
paper and implementation.

If you use this repository, please cite:

```bibtex
@inproceedings{zabergja2026endtoend,
    title={End-to-End Compression for Tabular Foundation Models},
    author={Guri Zab{\"e}rgja and Rafiq Kamel and Arlind Kadra and Christian Frey and Josif Grabocka},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=84mfkGDxYh}
}
```
