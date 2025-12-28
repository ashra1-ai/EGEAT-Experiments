# Kaggle Setup Guide for EGEAT Notebook

## Quick Start

1. **Upload the notebook** to Kaggle
2. **Add datasets** (if using GTSRB/Intel):
   - Click "Add Data" in the notebook
   - Search for "gtsrb-german-traffic-sign" or "intel-image-classification"
   - Add the dataset
3. **Set dataset** in the config cell:
   ```python
   DATASET = "CIFAR10"  # or "MNIST", "GTSRB", "INTEL"
   ```
4. **Run all cells**

## Environment Detection

The notebook automatically detects:
- **Kaggle**: Uses `/kaggle/working` for outputs
- **Colab**: Uses `/content` (fallback)
- **Local**: Uses current directory (fallback)

## Dataset Configuration

### Built-in Datasets (No setup needed)
- `"MNIST"` - Automatically downloads
- `"CIFAR10"` - Automatically downloads

### Kaggle Datasets (Requires adding dataset)
- `"GTSRB"` - German Traffic Sign Recognition
  - Add dataset: Search "gtsrb-german-traffic-sign"
  - Expected path: `/kaggle/input/gtsrb-german-traffic-sign/train` and `/test`
  
- `"INTEL"` - Intel Image Classification
  - Add dataset: Search "intel-image-classification"
  - Expected path: `/kaggle/input/intel-image-classification/train` and `/test`

## Output Files

All outputs are saved to `/kaggle/working/`:
- Model checkpoints
- Figures and visualizations
- CSV results
- Logs

These files are automatically included in your notebook output.

## GPU Settings

Kaggle provides:
- **P100 GPU** (free tier) - Sufficient for CIFAR-10/MNIST
- **T4 GPU** (free tier) - Good for larger datasets
- **A100 GPU** (paid) - For very large models

The notebook automatically uses GPU if available.

## Memory Considerations

For Kaggle free tier (30GB RAM):
- **CIFAR-10**: Works fine with batch_size=128
- **MNIST**: Works fine with batch_size=256
- **GTSRB/Intel**: May need batch_size=64 if memory issues

To reduce memory:
```python
# In dataset config cell
batch_size = 64  # Reduce if OOM errors
```

## Common Issues

### Issue: "Dataset not found"
**Solution:** Make sure you've added the dataset via "Add Data" button

### Issue: "Out of memory"
**Solution:** 
- Reduce `batch_size` to 64 or 32
- Reduce `ensemble_size` from 5 to 3
- Use `--mixed-precision` if available

### Issue: "CUDA out of memory"
**Solution:**
- Reduce batch size
- Clear cache: `torch.cuda.empty_cache()`
- Restart kernel and run again

## Performance Tips

1. **Enable GPU**: Settings → Accelerator → GPU
2. **Use mixed precision**: Already enabled in training
3. **Save checkpoints**: Models saved to `/kaggle/working/`
4. **Download outputs**: All files in `/kaggle/working/` are downloadable

## Expected Runtime (Kaggle P100)

- **CIFAR-10 (20 epochs)**: ~2-3 hours
- **MNIST (20 epochs)**: ~30-45 minutes
- **GTSRB (20 epochs)**: ~3-4 hours

## Notebook Structure for Kaggle

The notebook is organized for easy Kaggle browsing:

1. **Setup & Config** - Environment detection, dataset selection
2. **Standard Evaluation** - Results table (what recruiters see first)
3. **Efficiency Metrics** - Parameter counts, training cost
4. **Algorithmic Diagnostics** - Deep analysis (for researchers)

All outputs are clearly labeled and saved to `/kaggle/working/`.

