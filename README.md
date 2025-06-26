## Overview

This is the public code repository for our work
[CP^2: Leveraging Geometry for Conformal Prediction via Canonicalization](https://www.arxiv.org/abs/2506.16189) presented at [UAI 2025](https://www.auai.org/uai2025/).


#### 📝 Abstract 
---

We study the problem of conformal prediction (CP) under geometric data shifts, where data samples are susceptible to transformations such as rotations or flips. While CP endows prediction models with post-hoc uncertainty quantification and formal coverage guarantees, their practicality breaks under distribution shifts that deteriorate model performance. To address this issue, we propose integrating geometric information --such as geometric pose-- into the conformal procedure to reinstate its guarantees and ensure robustness under geometric shifts. In particular, we explore recent advancements on pose canonicalization as a suitable information extractor for this purpose. Evaluating the combined approach across discrete and continuous shifts and against equivariant and augmentation-based baselines, we find that integrating geometric information with CP yields a principled way to address geometric shifts while maintaining broad applicability to black-box predictors.

---

## 📌 Acknowledgments

This repository builds heavily on the following projects:

- [**equiadapt**](https://github.com/arnab39/equiadapt):  
  The project structure and many core components (e.g. data pipelines, training utilities) are adapted from this repository.  
  Licensed under the [MIT License](https://github.com/arnab39/equiadapt/blob/main/LICENSE).

- [**TorchCP**](https://github.com/ml-stat-sustech/torchcp):  
  Conformal prediction logic and CP predictors are adapted or reused from this library.  
  Licensed under the [LGPL-3.0 License](https://www.gnu.org/licenses/lgpl-3.0.html).


## Repo structure
This repository extends [equiadapt](https://github.com/arnab39/equiadapt/tree/main) with conformal prediction functionality using components adapted from [TorchCP](https://github.com/ml-stat-Sustech/TorchCP). Below is an overview of the core folders and files:

<pre>
image/
├── checkpoints/                  # Pretrained model checkpoints for CIFAR-10 and CIFAR-100
│   └── cifar10/, cifar100/      # Organized by dataset and symmetry group (e.g. base, canon_c8, so2)

├── configs/                      # Hydra-compatible YAML configuration files
│   ├── canonicalization/        # Group-equivariant or steerable settings
│   ├── cp/, dataset/, experiment/, prediction/, wandb/, checkpoint/
│   └── default.yaml             # Main config that composes all others

├── cp/                           # Conformal prediction logic
│   ├── cp_pipelines.py          # Main pipelines to run CP, MCP and WCP trials
│   ├── cp_predictors.py         # CP predictor classes (Split CP, Weighted CP, Mondrian CP)
│   ├── weighting_schemes.py     # Weighting functions for WCP
│   └── cp_utils.py              # Utilities for partitioning and binning

├── prepare/                      # Data preprocessing utilities
│   └── cifar_data.py            # Loads and prepares CIFAR datasets

├── common/                       # Shared utility functions
│   └── utils.py                 # Math and logging helpers

├── model.py                      # Model wrapper for classification
├── model_utils.py                # Helper functions for model construction
├── train_utils.py                # Training, evaluation, and model loading routines
├── inference_utils.py            # Inference-time prediction logic

├── cp.py                         # CP main program
├── mondrian_cp.py                # Mondrian CP main program
├── weight_cp.py                  # Weighted CP main program
├── info_utils.py                 # Computes image metadata (entropy, color, label)
</pre>


### 🧠 Checkpoints

We provide pretrained model checkpoints [**here**](https://drive.google.com/drive/folders/16bgg6Z4KoMpgQ1Jwz-huYb3tHZm08CRf?usp=sharing), and recommend placing them in the folder:

checkpoints/[dataset_name]/

#### 📄 Naming conventions:
- `base_*`: Standard models (normal predictors), optionally trained with data augmentation.
- `base_c4.ckpt`, `base_c8.ckpt`: Models trained with C4 or C8 rotational symmetry (cyclic groups of order 4 or 8).
- `base_so2.ckpt`: Model trained with continuous rotational symmetry (SO(2)).
- `canon_*`: Models with canonicalization applied before classification. The postfix (e.g., `canon_c4`, `canon_c8`) indicates the canonicalization group.

These models are used throughout the conformal prediction pipelines and reproduce results in the experiments.

#### Still open questions?

If there are any problems you encounter which have not been addressed, please feel free to create an issue or reach out! 
