import torch
import argparse
from torch_dataset import EventsDataset
import uproot
from parser import parse
from additional_data import add_additional_data

desc="""
This utility create a pytorch dataset (defined in root2torch/torch_dataset.py) from root files.

The main attributes of the dataset are:

- data (dict). In this dictionary are stored pytorch tensors
- info (dict). In this dictionary (same keys of data dictionary) are stored info on the columns of the data tensors

- additional_info (dict). Additional information defined by the user.

## singlet_dict

Define in singlet_dict the branches that you want to add to the dataset.
It accept also indexes (e.g. Muon_pt[0])
NB: the root file objects has to padded before to be of a fixed lenght. Do it in the preselection phase.
    (Use the RDataFrame and then Snapshot to save the skimmed file)
Example:
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
  Add 3 entries to the data dictionary:
  dataset.data["Jet"].shape=(N_events x N_Jet x 6)
  dataset.data["Lepton"].shape=(N_events x 1 x 1)
  dataset.data["MET"].shape=(N_events x 1 x 1)
  dataset.info["Jet"]=["Jet_pt","Jet_phi","Jet_eta","Jet_btagDeepFlavB","Jet_btagDeepFlavCvB","Jet_btagDeepFlavCvL"]

## couple_dict
if you have some object of size (N_events,n^2) in the root file and you want to create a tensor of the size
(N_events,n,n,1), define it in the couple_dict

##triplet_dict
Same of couple_dict but for (N_events,n^3) shaped objects

"""
if __name__=="__main__":
    parser = argparse.ArgumentParser(description=desc,
                                  formatter_class=argparse.RawDescriptionHelpFormatter )

    parser.add_argument("-i", "--input", help="Input root file name")
    parser.add_argument("-o", "--output", help="Output dataset file name")
    parser.add_argument("-g", "--generator", help="Specify if the dataset is generated with madgraph or powheg")
    parser.add_argument("-l", "--label", help="Label of the dataset")
    args=parser.parse_args()
    
    root_file=args.input
    output_file=args.output
    label=int(args.label)
    generator=args.generator
    
    assert generator in ["madgraph","powheg"]
    
    f=uproot.open(root_file)
    tree=f["Events"]
    
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
    
    couple_dict={"Masses":["Masses"],
                 }
    #triplet_dict={"Jet_THadMass":["Jet_THadMass"],}
    triplet_dict={}
    
    additional=["LeptLabel","HadDecay","AdditionalPartons"]

    singlet=parse(tree,singlet_dict)
    couple=parse(tree,couple_dict,type="couple")
    triplet=parse(tree,triplet_dict,type="triplet")

    dataset=EventsDataset()
    for key in singlet.keys():
        dataset.add_data(key,singlet[key],singlet_dict[key])
      
    for key in couple.keys():
        dataset.add_data(key,couple[key],["LeptMasses","WHadMasses"])
        
    for key in triplet.keys():
        dataset.add_data(key,triplet[key],triplet_dict[key])
    
    dataset.add_data("label",label*torch.ones(singlet["Jet"].shape[0],1),["label"])

    dataset=add_additional_data(dataset,tree,additional_list=additional,generator=generator)
    dataset.add_additional_info("generator",generator)
  
    print("I'm saving the dataset to: ",output_file)
    torch.save(dataset,output_file)

