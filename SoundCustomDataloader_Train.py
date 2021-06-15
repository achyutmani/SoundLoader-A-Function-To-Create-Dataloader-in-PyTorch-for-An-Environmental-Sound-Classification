import os
import torch
import numpy as np 
import h5py
from torch.utils.data import dataset
import pandas as pd 
from PIL import Image
class Sound_Data_Train():
	def __init__(self,transform=None):
		self.annotations=np.load('ESC10TrainData.npy',allow_pickle=True) # Read The names of Training Signals 
		self.Label=np.load('ESC10TrainLabel.npy',allow_pickle=True)# Numpy File with Class Labels
		self.Label=np.array(self.Label)
		self.transform=transform
	def __len__(self):
		return len(self.annotations)
	def __getitem__(self,index):
		key=self.annotations[index]
		with h5py.File('ESC10.hdf5', 'r') as f: # H5.py file That Contains Spectrogram Features for Each Signals Present in Training File
			SG_Data = f[key][()]
			SG_Data=np.array(SG_Data)
			SG_Data=Image.fromarray(SG_Data)
			SG_Label= torch.from_numpy(np.array((self.Label[index])))
			ES_Data=SG_Data
			if self.transform:
				ES_Data=self.transform(ES_Data)
		return (ES_Data,SG_Label)# Return ES_Data(Spectrogram Image Feature and Class label)
