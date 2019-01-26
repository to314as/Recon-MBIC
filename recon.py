import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import compare_ssim as ssim
import helper
# Input data files are available in the "../input/" directory.
import os
print(os.listdir("/"))
# Any results you write to the current directory are saved as output.

#load images

data_dir = 'C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\brain'
transform = transforms.Compose([transforms.Resize(256),
                                #transforms.CenterCrop(256),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)
img = images[2][0]

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image
data_transforms = transforms.Compose([
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor()
])
maskset = datasets.ImageFolder('C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\mask\\spiral', transform=transform)
dataloader = torch.utils.data.DataLoader(maskset, batch_size=1, shuffle=True)
masks, labels = next(iter(dataloader))
helper.imshow(masks[0], normalize=False)
mask= masks[0][0].numpy()
#mask=mask=cv2.imread('C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\mask\\spiral_60.tif',0)

#s=img*(mask!=0).float()
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = (20*np.log(np.abs(fshift)))

s=fshift*(mask==0)
f_ishift = np.fft.ifftshift(s)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

fig, ax = plt.subplots(figsize=(20, 10))
plt.subplot(141),plt.imshow(img.numpy(), cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('unmasked k-space'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(20*np.log(np.abs(s)), cmap = 'gray')
plt.title('masked k-space'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(img_back, cmap = 'gray')
plt.title('ifft'), plt.xticks([]), plt.yticks([])
plt.show()

helper.visualize(helper.difference(img.numpy(),img_back))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
transform = transforms.Compose([transforms.Resize(256),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])
batch_size = 20
# load the training and test datasets
train_data = datasets.ImageFolder(root='C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\brain\\db_train',
                                    transform=transform)
test_data = datasets.ImageFolder(root='C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\brain\\db_valid',
                                   transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.conv4 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 1, 3, padding=1)
        

    def forward(self, x):
        # add layer, with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decoder 
        # upsample, followed by a conv layer, with relu activation function  
        # this function is called `interpolate` in some PyTorch versions
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv4(x))
        # upsample again, output should have a sigmoid applied
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.sigmoid(self.conv5(x))
        
        return x
    
    
 # define the NN architecture
class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)  # kernel_size=3 to get to a 7x7 image output
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
                ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied
        x = F.sigmoid(self.conv_out(x))
                
        return x

# initialize the NN
model = ConvDenoiser()  
model.load_state_dict(torch.load('autodenoiseUp.pt', map_location="cuda"))
    
# initialize the NN
#model = ConvAutoencoder()
print(model)
model.to(device)
model.train()


# specify loss function
criterion = nn.MSELoss()
# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 100
mask=cv2.imread('C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\mask\\spiral_60.tif',0)

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images, _ = data
        ## add undersampling
        noisy_imgs = helper.undersample(images,mask)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        ## forward pass: compute predicted outputs by passing *noisy* images to the model
        outputs = model(noisy_imgs.cuda())
        # calculate the loss
        loss = criterion(outputs.cuda(), images.cuda())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
            
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))


torch.save(model.state_dict(), 'autodenoiseUp.pt')
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

noisy_imgs = helper.undersample(images,mask)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)
# get sample outputs
output = model(noisy_imgs.cuda())
#noisy_images
noisy_imgs = noisy_imgs.numpy()
# prep images for display
images = images.numpy()

# output is resized into a batch of iages
output = output.view(batch_size, 1, 256, 256)
# use detach when it's an output that requires_grad
output = output.detach().cpu().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(25,4))
c=0
# input images on top row, reconstructions on bottom
for images, row in zip([images, noisy_imgs, output], axes):
    mseN = helper.mse(images[0], images[1])
    ssimN= ssim(images[0][0], images[1][0])
    print(c)
    c+=1
    print(mseN)
    print(ssimN)
    mseO = helper.mse(images[0], images[2])
    ssimO= ssim(images[0][0], images[2][0])
    print(mseO)
    print(ssimO)
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)