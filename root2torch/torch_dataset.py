from torch.utils.data import Dataset
import torch

class EventsDataset(Dataset):
    """Pytorch dataset based on dictionaries

    Args:
        Dataset (_type_): pytorch dataset
    """
    def __init__(self):
        """ data: dictionary of tensors
            info: dictionary of column info
            additional_info: dictionary of additional info
        """
        self.data={}
        self.info={}
        self.additional_info={}

    def __len__(self):
        """Return the length of the dataset

        Returns:
            _type_: int
        """
        key=list(self.data.keys())[0]
        return len(self.data[key])

    
    def __getitem__(self, *key_idx):
        """ Return the element of the data dictionar
            dataset[key,idx/slice]
        Returns:
            _type_: tensor
        """
        return self.data[key_idx[0][0]][key_idx[0][1]]
    
    def add_data(self,name,data,column_info):
        """Add data to the data dictionary

        Args:
            name (_type_):key of the data dictionary
            data (_type_): tensor data
            column_info (_type_): description of the data columns
        """
        self.data[name]=data
        self.info[name]=column_info
    
    def add_additional_info(self,name,data):
        """ Add auxiliary informations

        Args:
            name (_type_): key of the dictionary
            data (_type_): value of the dictionary
        """
        self.additional_info[name]=data
    
    def to(self,device):
        """Move the data to the device

        Args:
            device (_type_): pytorch device
        """
        for key in self.data.keys():
            self.data[key]=self.data[key].to(device)
    
    def slice(self,start,end):
        """Select a slice of the data (on the first index) for all the keys
        (It overrites the existing data)
        

        Args:
            start (_type_): start index
            end (_type_): final index
        """
        for key in self.data.keys():
            self.data[key]=self.data[key][start:end]
    
    def cat(self,dataset):
        for key in self.data.keys():
            self.data[key]=torch.cat((self.data[key],dataset.data[key]),dim=0)
            
    def ls(self):
        """Print the keys and the shape of the data dictionary
        """
        for key in self.data.keys():
            print(f"key: {key}, shape: {self.data[key].shape}")
        