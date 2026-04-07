# Relative Density Ratio Optimization for Stable and Statistically Consistent Model Alignment

## About
This is a pytorch implementation of the following paper [[arXiv]](https://arxiv.org/abs/2604.04410):

```
@misc{takahashi2026relativedensityratiooptimization,
      title={Relative Density Ratio Optimization for Stable and Statistically Consistent Model Alignment}, 
      author={Hiroshi Takahashi and Tomoharu Iwata and Atsutoshi Kumagai and Sekitoshi Kanai and Masanori Yamada and Kosuke Nishida and Kazutoshi Shinoda},
      year={2026},
      eprint={2604.04410},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.04410}, 
}
```

Please read [LICENCE.md](LICENCE.md) before reading or using the files.

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
