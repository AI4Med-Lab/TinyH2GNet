import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import glob

# === Step 1: Locate and load all .h5ad datasets ===
# base_dir = "/home/puneet/mk/data/cscc_dataset"
base_dir = "/home/puneet/mk/data/her2st_dataset"
h5ad_paths = sorted(glob.glob(os.path.join(base_dir, "*.h5ad")))
print(h5ad_paths)

if len(h5ad_paths) == 0:
    raise FileNotFoundError(f"No .h5ad files found in {base_dir}")

print(f"✅ Found {len(h5ad_paths)} h5ad files.")
os.makedirs("gene_outputs", exist_ok=True)

# === Step 2: Load all and find common genes ===
adatas = [sc.read_h5ad(p) for p in h5ad_paths]
print(f"📦 Loaded all datasets. Shapes: {[a.shape for a in adatas]}")

common_genes = set(adatas[0].var_names)
for adata in adatas[1:]:
    common_genes &= set(adata.var_names)
common_genes = sorted(list(common_genes))
print(f"🧬 Common genes across all datasets: {len(common_genes)}")

# === Step 3: Subset to common genes and concatenate all datasets ===
adatas = [adata[:, common_genes].copy() for adata in adatas]
adata_combined = ad.concat(adatas, join="inner", label="sample", index_unique=None)
print(f"🧫 Combined dataset shape: {adata_combined.shape}")

# === Step 4: Normalize and log-transform ===
sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)

# === Step 5: Compute top 1000 Highly Variable Genes (HVGs) ===
sc.pp.highly_variable_genes(adata_combined, flavor="seurat_v3", n_top_genes=1000)
hvg_genes = adata_combined.var[adata_combined.var["highly_variable"]].index.tolist()
print(f"⭐ Found {len(hvg_genes)} HVGs after combining all sections")

# Subset to only the top 1000 HVGs
adata_hvg = adata_combined[:, hvg_genes].copy()
print(f"✅ Subsetted AnnData to top {adata_hvg.n_vars} HVGs")

# === Step 6: Remove genes expressed in fewer than 1000 spots ===
expr_counts = np.array((adata_hvg.X > 0).sum(axis=0)).flatten()
low_expr_genes = adata_hvg.var_names[expr_counts < 1000].tolist()
high_expr_mask = expr_counts >= 1000
filtered_genes = adata_hvg.var_names[high_expr_mask].tolist()

print(f"⚠️ Genes expressed in <1000 spots: {len(low_expr_genes)}")
print(f"✅ Genes retained after filtering: {len(filtered_genes)}")

# Final filtered AnnData
adata_final = adata_hvg[:, filtered_genes].copy()
print(f"🎯 Final dataset shape: {adata_final.shape}")

# === Step 7: Save outputs ===
pd.Series(common_genes).to_csv("gene_outputs/common_genes.csv", index=False)
pd.Series(hvg_genes).to_csv("gene_outputs/top1000_hvg_genes.csv", index=False)
pd.Series(filtered_genes).to_csv("gene_outputs/final_filtered_genes.csv", index=False)
pd.Series(low_expr_genes).to_csv("gene_outputs/low_expression_genes.csv", index=False)

# adata_combined.write_h5ad("gene_outputs/adata_combined_all_sections.h5ad")
# adata_hvg.write_h5ad("gene_outputs/adata_top1000_hvg.h5ad")
# adata_final.write_h5ad("gene_outputs/adata_final_filtered.h5ad")

print("\n✅ All files saved in ./gene_outputs/")
print(f"  - common_genes.csv          ({len(common_genes)})")
print(f"  - top1000_hvg_genes.csv     ({len(hvg_genes)})")
print(f"  - low_expression_genes.csv  ({len(low_expr_genes)})")
print(f"  - final_filtered_genes.csv  ({len(filtered_genes)})")
print(f"  - adata_final_filtered.h5ad shape: {adata_final.shape}")
