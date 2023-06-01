import torch
import argparse
from torch_dataset import EventsDataset
import uproot
from parser import parse
from additional_data import add_additional_data

desc="""
Just from a yaml config files, skim the root files and convert it to torch tensors.

Usage: python root2torch.py <config.yaml>

config.yaml Example:
---
root_files:
  files:
    - ./try.root                        #Can be a list of files and/or directories
  branches:                             #Branches to be loaded
    - Jet_pt
    - Jet_eta
    - Muon_pt
    - Muon_eta
  save:                                 # Path and branches to be saved as root file
                                        #   after the preselection: can also be False
    path: try_cuts.root
    branches:
      - Jet_pt
      - Jet_eta
      - Muon_pt
      - Muon_eta

preselection:                           # Preselection to be applied to the root files
                                        #based on RDataFrames. It accepts also wildcards
  - type: Redefine
    name: Muon*
    cut: Muon*[Muon_pt>26 && abs(Muon_eta)<2.4]
  - type: Filter
    cut: Jet_pt[0]>20


tensor:                                 # Tensor to be created from the rootfiles
  jet_data:                             # Name of attribute the tensor.
    type: ragged                        # Type of tensor: ragged (N x jets x Branches)
                                        #   or flat (N*Jets x Branches)
    clip: 7                             # Clip the number of jets. Can also be False in
                                        #   the type: flat case
    missing_values: -1.                 # What value to put on missing values: not
                                        #   needed if clip is False
    branches:                           # Which branches to save in the attribute
      - Jet_eta
      - Jet_pt
  
  mu_data:
    type: ragged
    clip: 1
    missing_values: -1.
    branches:
      - Muon_pt
      - Muon_eta

"""
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Convert root files to torch tensors",
                                  formatter_class=argparse.RawDescriptionHelpFormatter )
    """
    parser.add_argument("-c", "--config", help="Input yaml config file")
    stream=open(args.config)
    config = yaml.load(stream, Loader=yaml.FullLoader)
    preselection_config=config["preselection"]
    tensor_config=config["tensor"]
    """
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
    
  #! PASS ALWAY VECTOR BEFORE SINGLE
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

