# SoundLoader: A Package To Create Dataloader in PyTorch for An Environmental Sound Classification
Import Classes from SoundCustomDataloader_Train.py and SoundCustomDataloader_Test.py to Get the Training and Test Dataset for Sound Classification. Forward this dataset as an argument to dataloader in PyTorch. <br>
Sample Example Code: <br>
1. import torch<br>
2. from torch.utils.data import DataLoader<br>
3. from torchvision.transforms import transforms<br>
4. from SoundCustomDataloader_Train import Sound_Data_Train<br>
5. from SoundCustomDataloader_Test import Sound_Data_Test<br>
6. train_transformations = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=[0.229])])<br>
7. batch_size=32<br>
8. Training_Data=Sound_Data_Train(transform=transform)<br>
9. Test_Data=Sound_Data_Test(transform=transform)<br>
10. Train_loader=DataLoader(dataset=Train_Data,batch_size=batch_size,shuffle=True) # Pytorch dataloader for Training <br>
11. Test_loader=DataLoader(dataset=Train_Data,batch_size=batch_size,shuffle=True) Pytorch dataloader for Test <br>

