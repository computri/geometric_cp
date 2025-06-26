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
├── checkpoints/                 # Pretrained model checkpoints for CIFAR-10 and CIFAR-100
│   └── cifar10/, cifar100/      # Organized by dataset and symmetry group (e.g. base, canon_c8, so2)

├── configs/                     # Hydra-compatible YAML configuration files
│   ├── canonicalization/        # Group-equivariant or steerable settings
│   ├── cp/, dataset/, experiment/, prediction/, wandb/, checkpoint/
│   └── default.yaml             # Main config that composes all others

├── cp/                          # Conformal prediction logic
│   ├── cp_pipelines.py          # Main pipelines to run CP, MCP and WCP trials
│   ├── cp_predictors.py         # CP predictor classes (Split CP, Weighted CP, Mondrian CP)
│   ├── weighting_schemes.py     # Weighting functions for WCP
│   └── cp_utils.py              # Utilities for partitioning and binning

├── prepare/                      # Data preprocessing utilities
│   └── cifar_data.py             # Loads and prepares CIFAR datasets

├── common/                       # Shared utility functions
│   └── utils.py                  # Math and logging helpers

├── model.py                      # Model wrapper for classification
├── model_utils.py                # Helper functions for model construction
├── train_utils.py                # Training, evaluation, and model loading routines
├── inference_utils.py            # Inference-time prediction logic

├── cp.py                         # CP main program
├── mondrian_cp.py                # Mondrian CP main program
├── weight_cp.py                  # Weighted CP main program
├── info_utils.py                 # Computes image metadata (entropy, color, label)
</pre>

## 🔧 Setup

1. Clone the repository
```
git clone https://github.com/computri/geometric_cp.git
cd your-repo-name
```

2. Set environment variables, for example: 

```
export HYDRA_JOBS='./hydra_jobs'
export WANDB_ENTITY='computri'
export WANDB_PROJECT='cp2'
export WANDB_CACHE_DIR='./wandb_cache'
export DATA_PATH='./data'
export CHECKPOINT_PATH='./checkpoints'
```

3. Create and activate the Conda environment

```
conda env create -f environment.yml
conda activate cp2
```

4. Download model checkpoints

We provide pretrained model checkpoints [**here**](https://drive.google.com/drive/folders/16bgg6Z4KoMpgQ1Jwz-huYb3tHZm08CRf?usp=sharing), and recommend placing them in the folder:

checkpoints/[dataset_name]/

#### 📄 Naming conventions:
- `base_*`: Standard models (normal predictors), optionally trained with data augmentation.
- `base_c4.ckpt`, `base_c8.ckpt`: Models trained with C4 or C8 rotational symmetry (cyclic groups of order 4 or 8).
- `base_so2.ckpt`: Model trained with continuous rotational symmetry (SO(2)).
- `canon_*`: Models with canonicalization applied before classification. The postfix (e.g., `canon_c4`, `canon_c8`) indicates the canonicalization group.

These models are used throughout the conformal prediction pipelines and reproduce results in the experiments.

⸻


## 🚀 Usage

This project supports running experiments from the paper using different conformal prediction schemes. All experiments assume environment variables and checkpoints have been properly set up (see [Setup](#-setup)).

---

### 🔁 Robustness Experiments (Table 1)

Use `cp.py` for evaluating robustness under different canonicalization settings (i.e. Table 3, 6-9):

```
python cp.py \
  canonicalization=identity \
  cp.score_fn=APS \
  dataset.dataset_name=cifar100 \
  experiment.inference.num_rotations=8 \
  experiment.run_mode=test \
  cp.alpha=0.05 \
  checkpoint.checkpoint_path=./checkpoints \
  checkpoint.checkpoint_name=base \
  hydra.job.chdir=False
```

📌 Notes:
-	Use canonicalization=identity with checkpoint_name=base*
-	Use canonicalization=group_equivariant with checkpoint_name=canon_*
-	Vary experiment.inference.num_rotations (e.g. 4, 8) to produce different table entries

⸻

🧩 Mondrian Conformal Prediction (Table 2)

Use `mondrian_cp.py` for class-conditional and shifted data settings:

```
python mondrian_cp.py \
  canonicalization=group_equivariant \
  cp.score_fn=APS \
  dataset.dataset_name=cifar10 \
  experiment.inference.num_rotations=8 \
  experiment.run_mode=test \
  cp.alpha=0.1 \
  checkpoint.checkpoint_path=./checkpoints \
  checkpoint.checkpoint_name=canon_c8 \
  hydra.job.chdir=False \
  cp.num_resamples=100 \
  cp.canon_cutoff=0.90 \
  cp.mondrian.partitioning=label \
  dataset.cp_experiments.class_conditional_joint=1 \
  dataset.cp_experiments.shift_type=dirac
```

📌 Vary the following for different conditions:
-	canonicalization and checkpoint_name (e.g. identity + base_*, or group_equivariant + canon_*)
-	cp.mondrian.partitioning: e.g. label, entropy, color.
-	dataset.cp_experiments.shift_type: e.g. dirac, uniform, var-gauss

📊 Plotting results:
You can generate the relevant plots using `plots/mondrian_cp.ipynb`


⸻

⚖️ Weighted Conformal Prediction (Table 3)

Run weight_cp.py for weighted CP:

```
python weight_cp.py \
  canonicalization=group_equivariant \
  checkpoint.checkpoint_name=canon_c4 \
  checkpoint.checkpoint_path=./checkpoints \
  cp.alpha=0.05 \
  cp.num_resamples=10 \
  cp.score_fn=APS \
  cp.similarity_type=inverse_kl \
  cp.weight_lambda=1 \
  cp.weight_pow=0.5 \
  dataset.dataset_name=cifar10 \
  experiment.inference.num_rotations=4 \
  hydra.job.chdir=False \
  cp.weighted_cp.shift_augment=continuous_c4
```

📌 Notes:
-	Calibration data is automatically aligned with the canonicalizer (e.g. canon_c4 → c4 augment)
-	Use cp.weighted_cp.shift_augment to simulate test-time shift beyond the calibration distribution


#### Still open questions?

If there are any problems you encounter which have not been addressed, please feel free to create an issue or reach out! 
