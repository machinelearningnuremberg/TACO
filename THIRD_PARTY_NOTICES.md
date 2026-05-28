# Third-Party Notices

This repository includes TACO-original code and code derived from third-party
projects. Third-party components retain their original copyright notices,
licenses, and attribution requirements. Full copies of the third-party license
texts are bundled in the `LICENSES/` directory.

## Released Model Artifact Naming

Released trained model artifacts should be named with `TabPFN` as the first
token, for example:

- `TabPFN-TACO`
- `TabPFN-POT`
- `TabPFN-TACO-stage3-step-81025.ckpt`
- `TabPFN-POT-stage3-step-81025.ckpt`

Any public model card, checkpoint download page, documentation page, or product
surface for released TACO/POT model artifacts should prominently display:

Built with PriorLabs-TabPFN

TACO checkpoints released by this project are trained from scratch. They do not
redistribute or use TabPFN or TabICL pretrained weights.

## PriorLabs TabPFN

Path: `src/taco/model/tabpfn_arch/**`, except for files that are individually
attributed elsewhere in this document or in their own file headers. In
particular, `src/taco/model/tabpfn_arch/misc/_sklearn_compat.py` is covered by
the sklearn-compat section below, and `taco_classifier.py` and `shim_model.py`
are TACO-original code under the repository's main license.

This code is derived from and modifies TabPFN source code. Files modified by the
TACO contributors carry an in-file notice to that effect, in addition to the
retained upstream copyright notice.

- Upstream project: https://github.com/PriorLabs/TabPFN
- Upstream copyright: Copyright (c) Prior Labs GmbH 2025
- License: Prior Labs License, Apache 2.0 with additional attribution provision
- License URL: https://github.com/PriorLabs/TabPFN/blob/main/LICENSE
- Bundled license text: `LICENSES/PriorLabs-TabPFN-LICENSE.txt`

The TabPFN-derived code in this repository is distributed under the Prior Labs
License. The upstream license requires redistribution of the license, retention
of notices, prominent modification notices for modified files, and additional
attribution for distributed source, weights, products, services, or models built
with TabPFN source/model weights/outputs.

This repository does not include TabPFN pretrained checkpoint files.

## TabICL

Paths:

- `src/taco/prior/**`
- `src/taco/train/**`
- `scripts/train_stage1_*.sh`

Portions of the training and prior-generation code are derived from TabICL
pretraining code.

- Upstream project: https://github.com/soda-inria/tabicl
- Upstream copyright: Copyright (c) 2025, Soda team @ Inria
- License: BSD 3-Clause License
- License URL: https://github.com/soda-inria/tabicl/blob/main/LICENSE
- Bundled license text: `LICENSES/TabICL-LICENSE.txt`

This repository does not include TabICL pretrained checkpoint files.

## sklearn-compat

Path: `src/taco/model/tabpfn_arch/misc/_sklearn_compat.py`

This file is vendored from `sklearn-compat` to support multiple scikit-learn
versions without adding a runtime dependency on the `sklearn-compat` package.

- Upstream project: https://github.com/sklearn-compat/sklearn-compat
- Upstream documentation: https://sklearn-compat.readthedocs.io/en
- Vendored version: 0.1.3
- Upstream copyright: Copyright (c) 2024 Guillaume Lemaitre and Adrin Jalali
- License: BSD 3-Clause License
- License URL: https://github.com/sklearn-compat/sklearn-compat/blob/main/LICENSE
- Bundled license text: `LICENSES/sklearn-compat-LICENSE.txt`