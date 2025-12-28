# Kaggle Quick Start - EGEAT Notebook

## 🚀 3-Step Setup

### Step 1: Upload Notebook
- Upload `EGEAT_colab_heavy (1).ipynb` to Kaggle
- Or create new notebook and copy cells

### Step 2: Enable GPU (Recommended)
- Click **Settings** (gear icon)
- Set **Accelerator** → **GPU T4 x1** (free tier)
- Save settings

### Step 3: Run
- Click **Run All** or run cells sequentially
- All outputs saved to `/kaggle/working/`

## 📊 Dataset Options

### Option A: Built-in (No setup)
```python
DATASET = "CIFAR10"  # or "MNIST"
```
- Works immediately, no data needed
- Downloads automatically

### Option B: Kaggle Datasets
```python
DATASET = "GTSRB"  # or "INTEL"
```
- Click **"Add Data"** button
- Search: `gtsrb-german-traffic-sign` or `intel-image-classification`
- Add dataset
- Run notebook

## ⚡ Performance Tips

1. **GPU**: Essential for training (2-3 hours vs 10+ hours on CPU)
2. **Batch Size**: Reduce to 64 if OOM errors occur
3. **Epochs**: Start with 10-20 epochs for testing
4. **Mixed Precision**: Already enabled (faster training)

## 📁 Output Files

All saved to `/kaggle/working/`:
- `robustness_comparison.png` - Main results chart
- `training_cost_comparison.png` - Epoch comparison
- Model checkpoints (`.pt` files)
- CSV results
- All other visualizations

**Download**: Click "Output" tab → Download files

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `batch_size=64` in dataset config |
| Dataset Not Found | Add dataset via "Add Data" button |
| Slow Training | Enable GPU accelerator |
| CUDA Error | Restart kernel, enable GPU, run again |

## 📈 Expected Results

**CIFAR-10 (20 epochs, GPU T4):**
- Training time: ~2-3 hours
- PGD-20 Accuracy: ~0.27-0.31
- All figures generated automatically

**MNIST (20 epochs, GPU T4):**
- Training time: ~30-45 minutes  
- PGD-20 Accuracy: ~0.91-0.94

## ✅ Checklist Before Running

- [ ] Notebook uploaded to Kaggle
- [ ] GPU accelerator enabled
- [ ] Dataset selected (CIFAR10/MNIST/GTSRB/INTEL)
- [ ] If using GTSRB/INTEL: Dataset added via "Add Data"
- [ ] Ready to run!

The notebook automatically detects Kaggle and configures paths correctly.

