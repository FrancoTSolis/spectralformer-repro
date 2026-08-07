# Directly comparable baseline reruns

This directory isolates new baseline artifacts from all historical and proposed
model checkpoints.

## Data contract

Run once:

```bash
python prepare.py
```

The generated split files are the source of truth. See `PROVENANCE.md` for
method repositories, commits, and protocol details.

## MGN / MGN-NI

```bash
python train_mgn.py --dataset cylinder --variant mgn --gpu 0 --steps 10000
python train_mgn.py --dataset cylinder --variant mgn-ni --gpu 1 --steps 10000
```

Valid datasets are `cylinder`, `multiphase`, `we1`, and `e1`. Every run trains,
selects on validation data, performs the complete held-out rollout, and writes
one JSON file under `results/`.

## ViT-SEA

The official SEA checkout is trained through matched `*_retrain.py` configs.
After training:

```bash
python eval_vit_sea.py --dataset cylinder --gpu 0
```

## Audit

```bash
python aggregate.py
python aggregate.py --require-complete
```

The strict command fails until every required method/dataset result exists and
passes trajectory-count, horizon, and metric checks.
