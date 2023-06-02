# root2torch
This utility create a pytorch dataset (defined in root2torch/torch_dataset.py) from root files.

**Usage**: *python root2torch.py -i root_file.root -o dataset.pt -l 0 -g madgraph|powheg*

-i: root input file

-o: savepath of the final torch dataset

-l: label that will be added to each entry of dataset.data["label"]

-g: generator ("madgraph"|"powheg") (You can probably put one randomly, I use them in custom defined functions in additional_data.py)

The main attributes of the torch dataset are:

- data (dict). In this dictionary are stored pytorch tensors
- info (dict). In this dictionary (same keys of data dictionary) are stored info on the columns of the data tensors

- additional_info (dict). Additional information defined by the user. 

You can define functions in additional_data.py to define new objects and save where you want.

## singlet_dict

Define in singlet_dict the branches that you want to add to the dataset.
It accept also indexes (e.g. Muon_pt[0])

**NB: the root file objects has to padded before to be of a fixed lenght. 
Do it in the preselection phase.
    (Use the RDataFrame and then Snapshot to save the skimmed file)**
    
Example:
```python
    singlet_dict={"Jet":["Jet_pt",
                    "Jet_phi",
                    "Jet_eta",
                    "Jet_btagDeepFlavB",
                    "Jet_btagDeepFlavCvB",
                    "Jet_btagDeepFlavCvL"],
              "Lepton":["Muon_pt[0]",
                    "Muon_phi[0]",
                    "Muon_eta[0]"],
              "MET":["MET_pt",
                    "MET_phi",
                    "MET_eta"],
              }
  #Results:
  dataset.data["Jet"].shape=(N_events x N_Jet x 6)
  dataset.data["Lepton"].shape=(N_events x 1 x 1)
  dataset.data["MET"].shape=(N_events x 1 x 1)
  dataset.info["Jet"]=["Jet_pt","Jet_phi","Jet_eta","Jet_btagDeepFlavB","Jet_btagDeepFlavCvB","Jet_btagDeepFlavCvL"]
```

## couple_dict
if you have some object of size (N_events,n^2) in the root file and you want to create a tensor of the size
(N_events,n,n,1), define it in the couple_dict

## triplet_dict
Same of couple_dict but for (N_events,n^3) shaped objects
