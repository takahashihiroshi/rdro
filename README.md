# Relative Density Ratio Optimization for Stable and Statistically Consistent Model Alignment

## About


## Note
We recommend using 8× H100 GPUs or a comparable high-end GPU setup for training.


## Quickstart
```bash
uv sync -U --all-extras

# Align Llama-8B with RDRO on UF-G (alpha=0.39 is recommended)
accelerate launch --config_file accelerate/deepspeed_zero3.yaml main.py
```


## Hyperparameters
- Set `alpha` to the fraction of preferred data in the training set.
  - For `UF-G`, `alpha=0.39` is recommended.
