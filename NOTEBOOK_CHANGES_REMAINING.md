# Remaining Notebook Changes

## ✅ Completed Changes

1. ✅ Replaced `get_loaders()` with dataset-agnostic adapter
2. ✅ Added dataset switch cell after SAVE_DIR
3. ✅ Updated `make_model()` to be dataset-driven

## 🔄 Remaining Changes

### 4. Add PGD-50 Evaluation

**Location:** After the PGD-20 evaluation (around line 661)

**Add this code right after:**
```python
pgd_p = eval_adv_acc(pgd_model, test_loader, 'pgd', steps=20)
```

**Add:**
```python
# PGD-50 evaluation (credibility fix: evaluate with stronger attacks)
pgd50_e = eval_adv_acc(egeat_model, test_loader, 'pgd', steps=50)
pgd50_s = eval_adv_acc(egeat_soup,  test_loader, 'pgd', steps=50)
pgd50_p = eval_adv_acc(pgd_model,   test_loader, 'pgd', steps=50)
```

**Update the print statement from:**
```python
print(f"{'Model':<20} {'Clean Acc':<12} {'FGSM Acc':<12} {'PGD-20 Acc':<12}")
print("-"*70)
print(f"{'EGEAT Model':<20} {clean_e:.4f}      {fgsm_e:.4f}      {pgd_e:.4f}")
print(f"{'EGEAT Soup':<20} {clean_s:.4f}      {fgsm_s:.4f}      {pgd_s:.4f}")
print(f"{'PGD Model':<20} {clean_p:.4f}      {fgsm_p:.4f}      {pgd_p:.4f}")
```

**To:**
```python
print(f"{'Model':<20} {'Clean Acc':<12} {'FGSM Acc':<12} {'PGD-20 Acc':<12} {'PGD-50 Acc':<12}")
print("-"*70)
print(f"{'EGEAT Model':<20} {clean_e:.4f}      {fgsm_e:.4f}      {pgd_e:.4f}      {pgd50_e:.4f}")
print(f"{'EGEAT Soup':<20} {clean_s:.4f}      {fgsm_s:.4f}      {pgd_s:.4f}      {pgd50_s:.4f}")
print(f"{'PGD Model':<20} {clean_p:.4f}      {fgsm_p:.4f}      {pgd_p:.4f}      {pgd50_p:.4f}")
```

### 5. Remove Duplicate Robustness Surface

**Action:** Find and comment out or delete one of these duplicate robustness surface visualizations:
- `fig1_robustness_surface.png` (around line 691)
- Any "dashboard robustness surface" or "improvement surface" that shows the same metric

**Keep only ONE robustness surface plot** - the one that best represents the core robustness metric.

**Search for:** 
- `savefig("fig1_robustness_surface.png"`
- Any other robustness surface plots showing PGD-20 accuracy

**Recommendation:** Keep the first/main robustness surface and remove any duplicates that show interpolated versions of the same data.

---

## Summary

The three most critical changes (dataset adapter, dataset switch, model creation) are complete. The remaining two changes (PGD-50 evaluation and removing duplicate visuals) can be done manually in the notebook interface, as they are straightforward additions/deletions.

