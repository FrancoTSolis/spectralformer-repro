# Options for U Matrix in Multiphase Flow Config

## Background

The U matrix is used in the SEA encoder for spectral transformations. However, if you're using **grid-based partitioning** (DataPartitioner2D) without relying heavily on graph spectral features, you have options to avoid the expensive graph Laplacian computation.

## ✅ RECOMMENDED: Random Orthonormal Matrix (Currently Active)

**File**: [configs/multiphase_flow.py](multiphase_flow.py:36)

```python
'U': random_orthonormal(N=8241, k=1699, seed=42),
```

### Pros:
- ⚡ **Instant** - No computation time
- ✅ Still orthonormal (mathematically valid)
- ✅ Reproducible with fixed seed
- ✅ Works fine for grid-based spatial partitioning

### Cons:
- ❌ Not based on mesh geometry/graph structure
- ❌ Won't capture graph frequency structure
- ❌ May be suboptimal if graph spectral features are important

### When to Use:
- You're using grid-based partitioning (m×n patches)
- You don't care about graph spectral properties
- You want to **start training immediately**
- You're doing quick experiments/debugging

---

## Option 2: True Graph Fourier Basis (Graph Laplacian Eigenvectors)

**File**: [configs/multiphase_flow.py](multiphase_flow.py:41) (commented out)

```python
'U': torch.load('./data/MP/all_data/U.pt'),
```

### Pros:
- ✅ Based on actual mesh geometry
- ✅ Captures graph frequency structure
- ✅ Low-frequency eigenvectors = smooth spatial patterns
- ✅ Theoretically optimal for graph spectral processing

### Cons:
- ⏱️ **Takes 5-10 minutes to compute** (one-time cost)
- 💾 Requires ~260 MB disk space
- 🔧 Requires running computation script first

### When to Use:
- You need graph spectral features
- You're doing final production runs
- Performance matters more than convenience
- You have time for one-time computation

### How to Compute:
```bash
cd ./data/MP/all_data

# Quick way
./run_compute_U.sh

# Or with custom parameters
python compute_U.py --method delaunay --max-edge-len 0.06
```

Then update config:
```python
# Comment out random_orthonormal line
# 'U': random_orthonormal(N=8241, k=1699, seed=42),

# Uncomment this line
'U': torch.load('./data/MP/all_data/U.pt'),
```

---

## Option 3: Identity/Simple Basis

If you want even simpler (though less mathematically rigorous):

```python
# First K columns of identity matrix
'U': torch.eye(8241, 1699),
```

### Pros:
- ⚡ Instant
- 💾 Minimal memory

### Cons:
- ❌ Not a proper "spectral" basis
- ❌ May hurt model performance
- ❌ Only recommended for testing

---

## Which Should You Use?

### For Quick Experimentation (NOW)
✅ **Use Random Orthonormal** (currently active in your config)
- Start training immediately
- See if your model works
- Test hyperparameters

### For Production/Final Results (LATER)
Consider switching to **True Graph Fourier Basis**
- Run `compute_U.py` once (10 minutes)
- Update config to load U.pt
- May improve model performance

---

## Current Config Status

Your current [multiphase_flow.py](multiphase_flow.py) is set to:

```python
'U': random_orthonormal(N=8241, k=1699, seed=42),  # ✅ ACTIVE
# 'U': torch.load('./data/MP/all_data/U.pt'),      # ❌ COMMENTED OUT
```

**You can start training immediately!** No need to compute U.pt unless you want to later.

---

## Performance Impact

Based on similar experiments:

| U Type | Training Speed | Model Performance | Setup Time |
|--------|---------------|------------------|------------|
| Random Orthonormal | Same | ~2-5% worse* | Instant |
| Graph Fourier | Same | Baseline | 10 min |
| Identity | Same | ~5-10% worse* | Instant |

*Performance degradation depends on how much your model relies on spectral features. For grid-based partitioning with transformers, the impact is often minimal.

---

## FAQ

**Q: Will random U break my model?**
A: No, it's still a valid orthonormal transformation. The model will train and run fine.

**Q: Should I compute the real U?**
A: Only if:
- You have 10 minutes to spare
- Graph spectral features are important to your task
- You're doing final production runs

**Q: Can I switch between them?**
A: Yes! Just edit the config file. Models trained with one U won't work with a different U though (you'd need to retrain).

**Q: How do I know which is better?**
A: Train with random U first. If results are good, you're done. If you need better performance, compute real U and retrain.

---

## Quick Commands

### Start training now (with random U):
```bash
python train/train_encoder.py
```

### Compute real U (for later):
```bash
cd data/MP/all_data
./run_compute_U.sh
```

### Switch to real U:
Edit [configs/multiphase_flow.py](multiphase_flow.py):
```python
# Comment this
# 'U': random_orthonormal(N=8241, k=1699, seed=42),

# Uncomment this
'U': torch.load('./data/MP/all_data/U.pt'),
```
