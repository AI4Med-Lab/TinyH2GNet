import pandas as pd 
import scanpy as sc
import os

import numpy as np

# path = "/home/puneet/mk/data/cscc_dataset/"

# file = os.listdir("/home/puneet/mk/data/cscc_dataset/")

# print(file)

# print("===============================================================")
# file = [f for f in file if f[-2:] == "h5"]

# #['GSM4284320_patches.h5', 'GSM4284323_patches.h5', 'GSM4284322_patches.h5', 'GSM4284317_patches.h5', 'GSM4284327_patches.h5', 'GSM4284316_patches.h5', 'GSM4284326_patches.h5', 'GSM4284325_patches.h5', 'GSM4284324_patches.h5', 'GSM4284321_patches.h5', 'GSM4284319_patches.h5', 'GSM4284318_patches.h5']

# print(file)

# adata= os.listdir("/home/puneet/mk/data/cscc_dataset/")
# print("===============================================================")

# adata = [f for f in adata if f[-4:] == "h5ad"]

# #['GSM4284318_spots.h5ad', 'GSM4284317_spots.h5ad', 'GSM4284319_spots.h5ad', 'GSM4284324_spots.h5ad', 'GSM4284325_spots.h5ad', 'GSM4284326_spots.h5ad', 'GSM4284323_spots.h5ad', 'GSM4284316_spots.h5ad', 'GSM4284322_spots.h5ad', 'GSM4284327_spots.h5ad', 'GSM4284321_spots.h5ad', 'GSM4284320_spots.h5ad']

# print(adata)

# myadata = sc.read_h5ad(f"{path}/{adata[0]}")
# print(myadata)

# print(myadata.var)

# print(myadata.X)


# # df = pd.DataFrame(myadata.X, index=myadata.obs_names, columns=myadata.var_names)
# df = pd.DataFrame(myadata.X.toarray(), columns=myadata.var_names)


# print(df.head())



myfile = np.load('her2.npy', allow_pickle=True)

print(f'myfile {myfile}')
print(f'myfile {len(myfile)}')


# import h5py

# file_path = "/home/puneet/mk/data/cscc_dataset/GSM4284317_patches.h5"

# with h5py.File(file_path, "r") as f:
#     print("🔍 Keys in file:")
#     print(list(f.keys()))  # list of top-level datasets/groups

#     # If you know a key (for example "img"), show its shape and dtype
#     if "img" in f:
#         print(f"\n📸 Image dataset shape: {f['img'].shape}")
#         print(f"Data type: {f['img'].dtype}")