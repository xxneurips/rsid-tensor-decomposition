# Code release  --  RSID: Rank-Scheduled Iterative Distillation

Anonymous code release accompanying the NeurIPS 2026 submission

> *Stage-Leakage in Iterative Tensor Decomposition: A Methodology Note, an
> Algebraic Bound, and a CNN-vs-Transformer Asymmetry on ImageNet-1k*

(authors anonymous for double-blind review)

## What this code provides

This repository reproduces all experiments in the paper, including:

- RSID (Rank-Scheduled Iterative Distillation) on ResNet-18, ResNet-50, DeiT-Small, Swin-Tiny
- One-shot tensor decomposition baselines (Tucker, CP, Tensor Train, Tensor Ring)
- Structured channel pruning baseline + KD
- DepGraph (VainF/Torch-Pruning v1.6.1) baseline + KD
- MUSCO-style re-decomposition ablation
- Per-stage Frobenius approximation-error instrumentation (for the mechanism study)
- Singular value decay analysis (Conv2d vs. Linear)
- Long-schedule robustness check (5 vs. 30 epochs)
- Cross-dataset suite (CIFAR-10/100, Imagenette/Imagewoof, ImageNet-100, ImageNet-1k)

## Repository layout

```
.
├── src/
│   ├── decomposition/         # Tucker, CP, TT, TR, SVD-on-unfold, Tucker-on-Linear
│   ├── distillation/          # KD losses + trainer
│   ├── models/                # Model factory (ResNet, DeiT, Swin)
│   ├── utils/                 # data loaders, parquet streaming dataset, profiler
│   └── rsid.py                # RSID core
├── results/                   # Per-experiment JSONs (paired-seed accuracies, etc.)
├── run_imagenet1k_suite.py    # RN-18/RN-50 ImageNet-1k headline experiments
├── run_deit_imagenet1k.py     # DeiT-Small ImageNet-1k experiments
├── run_swin_t_rsid.py         # Swin-Tiny CIFAR-100 transformer experiments
├── run_mechanism_experiment.py# Per-stage epsilon and eta_i measurements
├── run_imagenet_resolution_suite.py # Imagenette/Imagewoof/ImageNet-100
├── run_budget_control.py      # CIFAR-100 budget control (150 epochs)
├── run_depgraph_baseline.py   # DepGraph baseline at matched compute
├── run_structured_pruning_baseline.py # Structured pruning + KD baseline
├── run_extra_seeds.py         # Extends multi-seed runs to n=5
├── run_experiment.py          # General one-shot / RSID / ablation runner
├── run_multiseed_and_tr.py    # Multi-seed + Tensor Ring evaluation
├── train_baseline.py          # FP32 teacher pre-training
├── make_scaling_law_figure.py # Figure 1 reproduction
├── sv_decay_analysis.py       # Appendix E.2 SV decay analysis (Figure 2)
├── compute_pvalues_5seed.py   # Statistical analysis (paired t-tests)
├── merge_and_recompute_n5.py  # Merge legacy n=3 with extra n=2 seeds
├── dl_parquet.py              # Hugging Face parquet ImageNet-1k downloader
├── cloud_download_imagenet.py # Cloud-runner ImageNet-1k downloader (parquet -> ImageFolder)
├── requirements.txt
└── README.md
```

## Reproducibility

Section B.5 of the paper lists exact hyperparameters. A single command reproduces
each experiment from a clean checkout. Random seeds (3, 7, 11, 42, 123) are pinned
via `torch.manual_seed`, `torch.cuda.manual_seed_all`, and `numpy.random.seed`,
with `cudnn.deterministic = True`.

## Requirements

- Python 3.10+
- PyTorch >= 2.0
- See `requirements.txt` for the full list (torchvision, tensorly, ptflops,
  pyarrow for parquet streaming, etc.)

## Headline experiments

```bash
# ResNet-18 / ResNet-50 on full ImageNet-1k (5 seeds RN-18, 3 seeds RN-50)
python run_imagenet1k_suite.py --device cuda:0 --seeds 3 7 11 42 123 \
    --output-json results/imagenet1k_suite_n5.json

# DeiT-Small on full ImageNet-1k (5 seeds)
python run_deit_imagenet1k.py --device cuda:0 --seeds 3 7 11 42 123 \
    --output-json results/deit_small_imagenet1k_n5.json

# Per-stage approximation-error mechanism experiment
python run_mechanism_experiment.py --device cuda:0 \
    --output-json results/mechanism_rn18_imagenet.json

# Singular-value decay analysis (Appendix E, Hypothesis 2 refutation)
python sv_decay_analysis.py
```

## Data

ImageNet-1k is loaded via the Hugging Face parquet mirror; see
`dl_parquet.py` and `src/utils/parquet_dataset.py` (an `IterableDataset` for
sequential row-group reads). CIFAR-10 / CIFAR-100 / Imagenette / Imagewoof /
ImageNet-100 use standard torchvision or HF loaders.

## License

Code: MIT.
Pretrained weights: subject to the respective dataset and architecture licenses.
