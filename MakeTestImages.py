import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torchvision import datasets, transforms
import helper

#load images
data_dir = 'C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\brain\\db_train'
transform = transforms.Compose([transforms.Resize(256),
                                #transforms.CenterCrop(256),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)
img = images[2][0]

maskset = datasets.ImageFolder('C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\mask', transform=transform)
dataloader = torch.utils.data.DataLoader(maskset, batch_size=9,shuffle=True)
masks, labels = next(iter(dataloader))
helper.imshow(masks[0], normalize=False)
print(labels[0][0])
mask=masks[0][0].numpy()

#s=img*(mask!=0).float()
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = (20*np.log(np.abs(fshift)))

s=fshift*(mask==0)
f_ishift = np.fft.ifftshift(s)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

fig, ax = plt.subplots(figsize=(20, 10))
plt.imshow(img_back, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])