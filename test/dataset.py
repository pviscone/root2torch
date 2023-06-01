#%%
import uproot
import torch
import numpy as np
import awkward as ak
from torch.utils.data import Dataset
import torch
import sys


f=uproot.open("../../Preselection_Skim/powheg/TTSemiLept_powheg_MuonCuts.root")
t=f["Events"]
#arr=t.arrays("Masses")["Masses"].to_numpy(allow_missing=False)
# %%
