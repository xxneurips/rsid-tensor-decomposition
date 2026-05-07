# RSID: Rank-Scheduled Iterative Distillation for Neural Network Compression
# ==========================================================================
# A comparative study of tensor decomposition methods (Tucker, CP, Tensor Train)
# combined with progressive knowledge distillation for CNN compression.
#
# Project structure:
#   src/
#   ├── decomposition/   - Tucker, CP, TT decomposition with CUDA acceleration
#   ├── distillation/    - Knowledge distillation losses and training loops
#   ├── models/          - Model loading, architecture configs
#   └── utils/           - Profiling, logging, data loading, rank scheduling
