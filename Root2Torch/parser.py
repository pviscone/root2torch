import torch
import numpy as np

def parse(tree,var_dict,type="single"):
    assert type in ["single","couple","triplet"]
    res={}
    for key in (var_dict.keys()):
        features=var_dict[key]
        for idx, feature in enumerate(features):
            feat_split=feature.split("[")
            print(feature)
            feature=feat_split[0]
            if len(feat_split)>1:
                index=int(feat_split[1].split("]")[0])
                new_column=torch.tensor(tree.arrays(feature)[feature][:,index,None].to_numpy()[:,:,None])
            else:
                new_column=torch.tensor(np.atleast_3d(tree.arrays(feature)[feature].to_numpy()))
                if type=="couple":
                    n=int(np.sqrt(new_column.shape[1]))
                    new_column=torch.reshape(new_column,(new_column.shape[0],n,n,1))
                if type=="triplet":
                    n=int(np.cbrt(new_column.shape[1]))
                    new_column=torch.reshape(new_column,(new_column.shape[0],n,n,n,1))
            if idx==0:
                res[key]=new_column
            else:
                if type=="single":
                    dim=2
                elif type=="couple":
                    dim=3
                elif type=="triplet":
                    dim=4
                res[key]=torch.cat((res[key],new_column),dim=dim)
    return res
