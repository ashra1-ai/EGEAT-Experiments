"""
Script to upgrade all 3D visualizations in the notebook to publication-quality smooth surfaces
"""
import json
import re

def upgrade_3d_visualizations(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Import statement to add at the beginning of visualization cells
    smooth_imports = """
from mpl_toolkits.mplot3d import Axes3D
try:
    from scipy.interpolate import RBFInterpolator
    HAS_RBF = True
except ImportError:
    from scipy.interpolate import griddata
    HAS_RBF = False
"""
    
    # Helper function for smooth interpolation
    smooth_interp_func = """
def smooth_interpolate(x_orig, y_orig, x_new, method='rbf'):
    \"\"\"Smooth interpolation for publication-quality surfaces\"\"\"
    if HAS_RBF and method == 'rbf' and len(x_orig) > 3:
        try:
            rbf = RBFInterpolator(x_orig.reshape(-1, 1), y_orig, 
                                 smoothing=0.1, kernel='thin_plate_spline')
            return rbf(x_new.reshape(-1, 1)).flatten()
        except:
            pass
    # Fallback to cubic interpolation
    from scipy.interpolate import interp1d
    f = interp1d(x_orig, y_orig, kind='cubic', fill_value='extrapolate')
    return f(x_new)
"""
    
    # Find and replace visualization cells
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Cell 6: EGEAT Training Progress
            if '# Visualize training progress' in source and 'fig, axes = plt.subplots(2, 2' in source:
                new_source = f"""{smooth_imports}
{smooth_interp_func}

# Publication-quality 3D Training Progress Visualization
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(DARK_THEME["figure.facecolor"])

epochs = np.array(egeat_history['epoch'])
loss_total = np.array(egeat_history['loss'])
loss_adv = np.array(egeat_history['adv_loss'])
geom_loss = np.array(egeat_history['geom_loss'])
soup_loss = np.array(egeat_history['soup_loss'])
val_acc = np.array(egeat_history['val_acc'])

# High-resolution grids for smooth surfaces
epoch_grid = np.linspace(epochs.min(), epochs.max(), 300)
loss_max = max(loss_total.max(), loss_adv.max(), geom_loss.max(), soup_loss.max()) * 1.1

# 1. Smooth Loss Evolution Surface
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
loss_grid = np.linspace(0, loss_max, 150)
E_grid, L_grid = np.meshgrid(epoch_grid, loss_grid)

Z_total = smooth_interpolate(epochs, loss_total, epoch_grid)
Z_adv = smooth_interpolate(epochs, loss_adv, epoch_grid)
Z_total = np.tile(Z_total, (len(loss_grid), 1))
Z_adv = np.tile(Z_adv, (len(loss_grid), 1))

surf1 = ax1.plot_surface(E_grid, L_grid, Z_total, cmap='viridis', alpha=0.88, 
                        edgecolor='none', linewidth=0, antialiased=True, shade=True,
                        rstride=3, cstride=3, vmin=Z_total.min(), vmax=Z_total.max())
surf2 = ax1.plot_surface(E_grid, L_grid, Z_adv, cmap='plasma', alpha=0.78, 
                        edgecolor='none', linewidth=0, antialiased=True, shade=True,
                        rstride=3, cstride=3, vmin=Z_adv.min(), vmax=Z_adv.max())

ax1.plot(epochs, np.zeros_like(epochs), loss_total, '-', label='Total Loss', 
        color=PALETTE["primary"], linewidth=5, alpha=0.95)
ax1.plot(epochs, np.zeros_like(epochs), loss_adv, '-', label='Adversarial Loss', 
        color=PALETTE["accent1"], linewidth=5, alpha=0.95)

ax1.view_init(elev=28, azim=42)
ax1.set_xlabel('Epoch', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax1.set_ylabel('', fontsize=14)
ax1.set_zlabel('Loss', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax1.set_title('Training Loss Evolution', fontsize=16, fontweight='bold', color=PALETTE["text"], pad=18)
ax1.legend(frameon=True, facecolor=DARK_THEME["axes.facecolor"], edgecolor=PALETTE["primary"], 
          labelcolor=PALETTE["text"], fontsize=11, loc='upper left', framealpha=0.95)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax1.yaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax1.zaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax1.tick_params(colors=PALETTE["text_secondary"], labelsize=11)
ax1.grid(True, alpha=0.25, linestyle='--')

# 2. Smooth Regularization Losses
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
Z_geom = smooth_interpolate(epochs, geom_loss, epoch_grid)
Z_soup = smooth_interpolate(epochs, soup_loss, epoch_grid)
Z_geom = np.tile(Z_geom, (len(loss_grid), 1))
Z_soup = np.tile(Z_soup, (len(loss_grid), 1))

surf3 = ax2.plot_surface(E_grid, L_grid, Z_geom, cmap='cool', alpha=0.88, 
                         edgecolor='none', linewidth=0, antialiased=True, shade=True,
                         rstride=3, cstride=3)
surf4 = ax2.plot_surface(E_grid, L_grid, Z_soup, cmap='winter', alpha=0.78, 
                         edgecolor='none', linewidth=0, antialiased=True, shade=True,
                         rstride=3, cstride=3)

ax2.plot(epochs, np.zeros_like(epochs), geom_loss, '-', label='Geometric Loss', 
        color=PALETTE["accent2"], linewidth=5, alpha=0.95)
ax2.plot(epochs, np.zeros_like(epochs), soup_loss, '-', label='Soup Loss', 
        color=PALETTE["accent3"], linewidth=5, alpha=0.95)

ax2.view_init(elev=28, azim=42)
ax2.set_xlabel('Epoch', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax2.set_ylabel('', fontsize=14)
ax2.set_zlabel('Loss', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax2.set_title('Regularization Loss Evolution', fontsize=16, fontweight='bold', color=PALETTE["text"], pad=18)
ax2.legend(frameon=True, facecolor=DARK_THEME["axes.facecolor"], edgecolor=PALETTE["primary"], 
          labelcolor=PALETTE["text"], fontsize=11, loc='upper left', framealpha=0.95)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax2.yaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax2.zaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax2.tick_params(colors=PALETTE["text_secondary"], labelsize=11)
ax2.grid(True, alpha=0.25, linestyle='--')

# 3. Smooth Validation Accuracy
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
acc_grid = np.linspace(0, 1, 150)
E_acc, A_grid = np.meshgrid(epoch_grid, acc_grid)

Z_acc = smooth_interpolate(epochs, val_acc, epoch_grid)
Z_acc = np.tile(Z_acc, (len(acc_grid), 1))

surf5 = ax3.plot_surface(E_acc, A_grid, Z_acc, cmap='viridis', alpha=0.92, 
                        edgecolor='none', linewidth=0, antialiased=True, shade=True,
                        rstride=3, cstride=3, vmin=0, vmax=1)

ax3.plot(epochs, np.zeros_like(epochs), val_acc, '-', 
        color=PALETTE["success"], linewidth=6, alpha=0.98)

ax3.view_init(elev=28, azim=42)
ax3.set_xlabel('Epoch', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax3.set_ylabel('', fontsize=14)
ax3.set_zlabel('Validation Accuracy', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax3.set_title('Validation Accuracy Evolution', fontsize=16, fontweight='bold', color=PALETTE["text"], pad=18)
ax3.set_zlim(0, 1.0)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.xaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax3.yaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax3.zaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax3.tick_params(colors=PALETTE["text_secondary"], labelsize=11)
ax3.grid(True, alpha=0.25, linestyle='--')

# 4. Smooth Loss Components Decomposition
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
cfg = EGEATConfig()
adv_comp = np.array(egeat_history['adv_loss'])
geom_comp = np.array(egeat_history['geom_loss']) * cfg.lambda_geom
soup_comp = np.array(egeat_history['soup_loss']) * cfg.lambda_soup
total_comp = adv_comp + geom_comp + soup_comp

comp_grid = np.linspace(0, 1, 100)
E_comp, C_grid = np.meshgrid(epoch_grid, comp_grid)

Z_adv_surf = smooth_interpolate(epochs, adv_comp, epoch_grid)
Z_geom_surf = smooth_interpolate(epochs, geom_comp, epoch_grid)
Z_soup_surf = smooth_interpolate(epochs, soup_comp, epoch_grid)
Z_adv_surf = np.tile(Z_adv_surf, (len(comp_grid), 1))
Z_geom_surf = np.tile(Z_geom_surf, (len(comp_grid), 1))
Z_soup_surf = np.tile(Z_soup_surf, (len(comp_grid), 1))

ax4.plot_surface(E_comp, C_grid, Z_adv_surf, cmap='viridis', alpha=0.88, 
                edgecolor='none', linewidth=0, antialiased=True, shade=True, rstride=3, cstride=3)
ax4.plot_surface(E_comp, C_grid, Z_adv_surf + Z_geom_surf, cmap='plasma', alpha=0.78, 
                edgecolor='none', linewidth=0, antialiased=True, shade=True, rstride=3, cstride=3)
ax4.plot_surface(E_comp, C_grid, Z_adv_surf + Z_geom_surf + Z_soup_surf, cmap='cool', alpha=0.68, 
                edgecolor='none', linewidth=0, antialiased=True, shade=True, rstride=3, cstride=3)

ax4.plot(epochs, np.zeros_like(epochs), adv_comp, '-', label='Adversarial', 
        color=PALETTE["accent1"], linewidth=5, alpha=0.95)
ax4.plot(epochs, np.zeros_like(epochs), adv_comp + geom_comp, '-', label='+ Geometric', 
        color=PALETTE["accent2"], linewidth=5, alpha=0.95)
ax4.plot(epochs, np.zeros_like(epochs), total_comp, '-', label='Total', 
        color=PALETTE["primary"], linewidth=5, alpha=0.95)

ax4.view_init(elev=28, azim=42)
ax4.set_xlabel('Epoch', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax4.set_ylabel('', fontsize=14)
ax4.set_zlabel('Loss Components', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax4.set_title('Loss Components Decomposition', fontsize=16, fontweight='bold', color=PALETTE["text"], pad=18)
ax4.legend(frameon=True, facecolor=DARK_THEME["axes.facecolor"], edgecolor=PALETTE["primary"], 
          labelcolor=PALETTE["text"], fontsize=11, loc='upper left', framealpha=0.95)
ax4.xaxis.pane.fill = False
ax4.yaxis.pane.fill = False
ax4.zaxis.pane.fill = False
ax4.xaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax4.yaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax4.zaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax4.tick_params(colors=PALETTE["text_secondary"], labelsize=11)
ax4.grid(True, alpha=0.25, linestyle='--')

plt.suptitle('EGEAT Training Progress Analysis', fontsize=24, fontweight='bold', color=PALETTE["text"], y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
savefig("fig_training_progress_3d.png", dpi=300)
plt.show()
"""
                nb['cells'][i]['source'] = new_source.split('\n')
                print(f"✓ Upgraded Cell {i}: EGEAT Training Progress")
            
            # Cell 7: PGD Training Progress
            elif '# Visualize PGD training progress' in source and 'fig, axes = plt.subplots(1, 2' in source:
                new_source = f"""{smooth_imports}
{smooth_interp_func}

# Publication-quality 3D PGD Training Progress
fig = plt.figure(figsize=(18, 8))
fig.patch.set_facecolor(DARK_THEME["figure.facecolor"])

epochs_pgd = np.array(pgd_history['epoch'])
loss_pgd = np.array(pgd_history['loss'])
val_acc_pgd = np.array(pgd_history['val_acc'])

epoch_grid = np.linspace(epochs_pgd.min(), epochs_pgd.max(), 300)

# 1. Smooth Loss Surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
loss_grid = np.linspace(0, loss_pgd.max() * 1.1, 150)
E_grid, L_grid = np.meshgrid(epoch_grid, loss_grid)

Z_loss = smooth_interpolate(epochs_pgd, loss_pgd, epoch_grid)
Z_loss = np.tile(Z_loss, (len(loss_grid), 1))

surf1 = ax1.plot_surface(E_grid, L_grid, Z_loss, cmap='viridis', alpha=0.92, 
                        edgecolor='none', linewidth=0, antialiased=True, shade=True,
                        rstride=3, cstride=3)

ax1.plot(epochs_pgd, np.zeros_like(epochs_pgd), loss_pgd, '-', 
        color=PALETTE["primary"], linewidth=6, alpha=0.98)

ax1.view_init(elev=28, azim=42)
ax1.set_xlabel('Epoch', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax1.set_ylabel('', fontsize=14)
ax1.set_zlabel('Training Loss', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax1.set_title('PGD Training Loss Evolution', fontsize=16, fontweight='bold', color=PALETTE["text"], pad=18)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax1.yaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax1.zaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax1.tick_params(colors=PALETTE["text_secondary"], labelsize=11)
ax1.grid(True, alpha=0.25, linestyle='--')

# 2. Smooth Validation Accuracy Surface
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
acc_grid = np.linspace(0, 1, 150)
E_acc, A_grid = np.meshgrid(epoch_grid, acc_grid)

Z_acc = smooth_interpolate(epochs_pgd, val_acc_pgd, epoch_grid)
Z_acc = np.tile(Z_acc, (len(acc_grid), 1))

surf2 = ax2.plot_surface(E_acc, A_grid, Z_acc, cmap='plasma', alpha=0.92, 
                        edgecolor='none', linewidth=0, antialiased=True, shade=True,
                        rstride=3, cstride=3, vmin=0, vmax=1)

ax2.plot(epochs_pgd, np.zeros_like(epochs_pgd), val_acc_pgd, '-', 
        color=PALETTE["success"], linewidth=6, alpha=0.98)

ax2.view_init(elev=28, azim=42)
ax2.set_xlabel('Epoch', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax2.set_ylabel('', fontsize=14)
ax2.set_zlabel('Validation Accuracy', fontsize=14, color=PALETTE["text"], fontweight='bold', labelpad=15)
ax2.set_title('PGD Validation Accuracy Evolution', fontsize=16, fontweight='bold', color=PALETTE["text"], pad=18)
ax2.set_zlim(0, 1.0)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax2.yaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax2.zaxis.pane.set_edgecolor(PALETTE["text_secondary"])
ax2.tick_params(colors=PALETTE["text_secondary"], labelsize=11)
ax2.grid(True, alpha=0.25, linestyle='--')

plt.suptitle('PGD Training Progress Analysis', fontsize=20, fontweight='bold', color=PALETTE["text"], y=1.02)
plt.tight_layout()
savefig("fig_pgd_training_3d.png", dpi=300)
plt.show()
"""
                nb['cells'][i]['source'] = new_source.split('\n')
                print(f"✓ Upgraded Cell {i}: PGD Training Progress")
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\n✓ Notebook '{notebook_path}' upgraded with publication-quality 3D visualizations!")

if __name__ == '__main__':
    upgrade_3d_visualizations('EGEAT_colab_heavy (1).ipynb')

