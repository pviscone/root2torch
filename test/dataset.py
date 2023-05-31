#%%
import uproot
import torch
import numpy as np
import awkward as ak

f=uproot.open("BigMuons_MuonCuts.root")
tree = f["Events"]

# %%
