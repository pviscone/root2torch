#%%

import torch
from torch_dataset import EventsDataset


#%%
#Load signal and background datasets
signal=torch.load("../../Preselection_Skim/signal/signal_MuonCuts.pt")
powheg=torch.load("../../Preselection_Skim/powheg/TTSemiLept_powheg_MuonCuts.pt")
signal.shuffle()
powheg.shuffle()
n_sig=len(signal)



#Select only muons in the background and add to the signal
lepton_powheg=powheg.mask(torch.abs(powheg.data["LeptLabel"])==13, retrieve=True)
powheg_nn=lepton_powheg.get_batch(0,n_sig*4)
nn=signal.cat(powheg_nn, retrieve=True)


#Add the rest of the mu background to the non muon background
other_lepton=lepton_powheg.get_batch(n_sig*4,-1)
non_lepton_powheg=powheg.mask(torch.abs(powheg.data["LeptLabel"])!=13, retrieve=True)
other_dataset=other_lepton.cat(non_lepton_powheg, retrieve=True)

#Shuffle the datasets
other_dataset.shuffle()
nn.shuffle()

#create train and test datasets
test_size=0.15
train_dataset=nn.get_batch(0,int((1-test_size)*len(nn)))
test_dataset=nn.get_batch(int((1-test_size)*len(nn)),-1)


torch.save(train_dataset,"../../Preselection_Skim/NN/train_Muons.pt")
torch.save(test_dataset,"../../Preselection_Skim/NN/test_Muons.pt")
torch.save(other_dataset,"../../Preselection_Skim/NN/TTSemilept_MuonCuts.pt")
