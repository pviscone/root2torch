#%%

import torch
from torch_dataset import EventsDataset


#%%
#Load signal and background datasets
signal=torch.load("../../Preselection_Skim/signal/signal_MuonCuts.pt")
semiLept=torch.load("../../Preselection_Skim/powheg/TTSemiLept_MuonCuts.pt")
diLept=torch.load("../../Preselection_Skim/diLept/TTdiLept_MuonCuts.pt")

signal.add_data("type",torch.ones(len(signal)),["1 signal, 0 semiLeptBkg, -1 diLeptBkg"])
semiLept.add_data("type",torch.zeros(len(semiLept)),["1 signal, 0 semiLeptBkg, -1 diLeptBkg"])
diLept.add_data("type",torch.ones(len(diLept))*-1,["1 signal, 0 semiLeptBkg, -1 diLeptBkg"])

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
train_dataset=train_signal.cat(train_diLept, retrieve=True)
train_dataset=train_dataset.cat(train_semiLept, retrieve=True)

test_signal=signal.get_batch(n_train,-1)
test_semiLept=semiLept.get_batch(n_train*3,n_train*3+3*len(test_signal))
test_diLept=diLept.get_batch(n_train,n_train+len(test_signal))
test_dataset=test_signal.cat(test_diLept, retrieve=True)
test_dataset=test_dataset.cat(test_semiLept, retrieve=True)

other_semilept=semiLept.get_batch(n_train*3+3*len(test_signal),-1)
other_diLept=diLept.get_batch(n_train+len(test_signal),-1)
others=other_semilept.cat(other_diLept, retrieve=True)

train_dataset.shuffle()
test_dataset.shuffle()
others.shuffle()

torch.save(train_dataset,"../../Preselection_Skim/NN/train_Muons.pt")
torch.save(test_dataset,"../../Preselection_Skim/NN/test_Muons.pt")
torch.save(others,"../../Preselection_Skim/NN/OtherBkg_Muons.pt")
torch.save(test_signal,"../../Preselection_Skim/NN/test_signal_Muons.pt")

# %%
