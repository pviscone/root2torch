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
            ak_array=tree.arrays(feature)[feature]
            if len(feat_split)>1:
                index=int(feat_split[1].split("]")[0])
                new_column=torch.tensor(ak_array[:,index,None].to_numpy()[:,:,None])
            else:
                shape0=len(ak_array)
                if ak_array.ndim==1:
                    new_column=ak_array.to_numpy()[:,None]
                    shape1=1
                else:
                    new_column=np.array(ak_array.layout.content)
                    shape1=ak_array.layout.offsets[1]

                new_column=torch.tensor(new_column)
                if type=="single":
                    new_column=torch.reshape(new_column,(shape0,shape1,1))
                elif type=="couple":
                    n=int(np.sqrt(shape1))
                    new_column=torch.reshape(new_column,(shape0,n,n,1))
                elif type=="triplet":
                    n=int(np.cbrt(shape1))
                    new_column=torch.reshape(new_column,(shape0,n,n,n,1))
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
