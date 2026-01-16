# Baseline Transformer

A compact transformer language model for training on PTB and WikiText-2 datasets.

## Setup

```bash
pip install -r requirements.txt
cd datasets && python download_datasets.py
```

## Training

```bash
# Light version (synthetic data, faster iteration)
python -m train.baseline.train_baseline_light --optimizer adamw --epochs 2 --lr 3e-4 --weight_decay 0.01

# Full training
python -m train.baseline.train_baseline --dataset wikitext2 --optimizer adamw --n_layers 4 --lr 0.003 --weight_decay 0.1 --batch_size 16 --epochs 10 --seq_len 128 --max_vocab_size 4096

# Hyperparameter sweeps
python -m train.baseline.sweep_baseline
python -m train.baseline.sweep_baseline_light
```

## Evaluation

```bash
python -m evals.generate_text --checkpoint_path runs/baseline_wikitext2.pt --prompt "the dog"
```
