import torch
import awkward as ak
import numpy as np
from torch_dataset import EventsDataset

def LeptLabel(tree):
    pdgId=tree["LHEPart_pdgId"].array()
    lept_label=torch.ones(ak.num(pdgId,axis=0),1)
    
    MuonMask=ak.sum(np.abs(pdgId)==13,axis=1).to_numpy()
    ElectronMask=ak.sum(np.abs(pdgId)==11,axis=1).to_numpy()
    TauMask=ak.sum(np.abs(pdgId)==15,axis=1).to_numpy()
    
    lept_label[MuonMask==1]*=13
    lept_label[MuonMask==2]*=13
    lept_label[ElectronMask==1]*=11
    lept_label[ElectronMask==2]*=11
    lept_label[TauMask==1]*=15
    lept_label[TauMask==2]*=15
    
    return lept_label

def HadDecay(tree,generator):
    assert generator in ["powheg","madgraph"]
    pdgId=tree["LHEPart_pdgId"].array()
    if generator=="powheg":
        had_decay=pdgId[:,4:]
        had_decay=had_decay[np.abs(had_decay)<5]
        had_decay=torch.tensor(had_decay.to_numpy(),dtype=int)
        return had_decay
    elif generator=="madgraph":
        return None
    
def AdditionalPartons(tree,generator):
    assert generator in ["powheg","madgraph"]
    pdgId=tree["LHEPart_pdgId"].array()
    n_partons=ak.num(pdgId,axis=1).to_numpy()
    additional_partons=np.zeros(len(pdgId))
    if generator=="powheg":
        additional_partons[n_partons==9]=pdgId[n_partons==9,2].to_numpy()
    elif generator=="madgraph":
        additional_partons=ak.pad_none(tree["LHEPart_pdgId"].array(),11,clip=True,axis=1)
        additional_partons=ak.fill_none(additional_partons,0).to_numpy()[:,8:]
    return torch.tensor(additional_partons,dtype=int)


def add_additional_data(dataset, tree,additional_list,generator=None):
    assert generator in ["powheg","madgraph"]
    for info in additional_list:
        print(info)
        assert info in ["LeptLabel","HadDecay","AdditionalPartons"]
        if info=="LeptLabel":
            dataset.add_data(info,LeptLabel(tree),[info])
        elif info=="HadDecay":
            dataset.add_data(info,HadDecay(tree,generator=generator),[info])
        elif info=="AdditionalPartons":
            dataset.add_data(info,AdditionalPartons(tree,generator=generator),[info])
    return dataset
            