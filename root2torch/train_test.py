#%%

import torch
from torch_dataset import EventsDataset


#%%
#Load signal and background datasets
signal=torch.load("../../Preselection_Skim/signal/signal_MuonCuts.pt")
semiLept=torch.load("../../Preselection_Skim/powheg/TTSemiLept_MuonCuts.pt")
diLept=torch.load("../../Preselection_Skim/diLept/TTdiLept_MuonCuts.pt")
signal.shuffle()
semiLept.shuffle()
diLept.shuffle()
#%%
n_sig=len(signal)
n_train=int(n_sig*0.85)

#Build training and test datasets
train_signal=signal.get_batch(0,n_train)
train_semiLept=semiLept.get_batch(0,n_train*3)
train_diLept=diLept.get_batch(0,n_train)
train_dataset=signal.cat(train_diLept, retrieve=True)
train_dataset=train_dataset.cat(train_semiLept, retrieve=True)
train_dataset.shuffle()

test_signal=signal.get_batch(n_train,-1)
test_semiLept=semiLept.get_batch(n_train*3,-1)
test_diLept=diLept.get_batch(n_train,-1)
test_dataset=signal.cat(test_diLept, retrieve=True)
test_dataset=test_dataset.cat(test_semiLept, retrieve=True)
test_dataset.shuffle()

torch.save(train_dataset,"../../Preselection_Skim/NN/train_Muons.pt")
torch.save(test_dataset,"../../Preselection_Skim/NN/test_Muons.pt")


# %%
