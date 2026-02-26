#Jai Shri Ganesh


import pandas as pd
import os
import numpy as np 
import scanpy as sc
import h5py

path = "/home/puneet/mk/data/cscc_dataset/224"

files = os.listdir(path)

print(files)

# patch_data = sc.read_h5ad(f"{path}/{files[0]}")

# print(patch_data)

with h5py.File(f"{path}/{files[0]}") as f:
	print(f.keys())
	print(f["x"])
	print(f["y"])

