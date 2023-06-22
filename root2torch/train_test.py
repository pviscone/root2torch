#%%

import torch
from torch_dataset import EventsDataset


#%%
#Load signal and background datasets
signal_train=torch.load("../../Preselection_Skim/NN/train/torch/signal_train_MuonCuts.pt")
semiLept_train=torch.load("../../Preselection_Skim/NN/train/torch/TTSemiLept_train_MuonCuts.pt")
diLept_train=torch.load("../../Preselection_Skim/NN/train/torch/TTdiLept_train_MuonCuts.pt")


signal_test=torch.load("../../Preselection_Skim/NN/test/torch/signal_test_MuonCuts.pt")
semiLept_test=torch.load("../../Preselection_Skim/NN/test/torch/TTSemiLept_test_MuonCuts.pt")
diLept_test=torch.load("../../Preselection_Skim/NN/test/torch/TTdiLept_test_MuonCuts.pt")

train_dataset=signal_train.cat(diLept_train, retrieve=True)
train_dataset=train_dataset.cat(semiLept_test, retrieve=True)

test_dataset=signal_test.cat(diLept_test, retrieve=True)
test_dataset=test_dataset.cat(semiLept_test, retrieve=True)
train_dataset.shuffle()
test_dataset.shuffle()


torch.save(train_dataset,"../../Preselection_Skim/NN/train_Muons.pt")
torch.save(test_dataset,"../../Preselection_Skim/NN/test_Muons.pt")

# %%
