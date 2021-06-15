import os
import torch
import numpy as np 
import h5py
from torch.utils.data import dataset
import pandas as pd 
from PIL import Image
class Sound_Data_Train():
	def __init__(self,transform=None):
		self.annotations=np.array(pd.read_csv("TrainFile.csv"))# Read The CSV file WIth Names of Training Signals and Class Labels
		self.transform=transform
	def __len__(self):
		return len(self.annotations)
	def __getitem__(self,index):
		key=self.annotations[index,0]
		#print(index)
		with h5py.File('LDCASE2019.hdf5', 'r') as f: # H5.py file That Contains Spectrogram Features for All The Training Signals
			SG_Data = f[key][()]
			SG_Data=Image.fromarray(SG_Data)
			SG_Label= torch.from_numpy(np.array(self.annotations[index,1]))
			if self.transform:
				SG_Data=self.transform(SG_Data)
		return (SG_Data,SG_Label)	

		
		
