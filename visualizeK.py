import helper
import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch
from torchvision import datasets, transforms


transform = transforms.Compose([transforms.Resize(256),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])

img=cv2.imread('C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\brain\\db_train\\Original\\001.png',0)
mask=cv2.imread('C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\mask\\spiral_60.tif',0)


## add undersampling
noisy_imgs = helper.undersampleOne(img,mask)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = (20*np.log(np.abs(fshift)))
s=fshift*(mask==0)
ims=20*np.log(np.abs(s))
f_ishift = np.fft.ifftshift(s)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

rows,cols = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
img_rot = np.rot90(img)
img_180=np.rot90(img_rot)

img_rot=[[1,1,1,0],[1,1,1,0],[1,1,1,0],[1,1,1,0]]
img_180=np.rot90(img_rot)

f_90 = np.fft.fft2(img_rot)
fshift_90 = np.fft.fftshift(f_90)
magnitude_spectrum_90 = (20*np.log(np.abs(fshift_90)))

f_180 = np.fft.fft2(img_180)
fshift_180 = np.fft.fftshift(f_180)
magnitude_spectrum_180 = (20*np.log(np.abs(fshift_180)))

helper.visualize(helper.difference(img,img_back))
#helper.visualize(helper.difference(magnitude_spectrum,magnitude_spectrum_180))

fig, ax = plt.subplots(figsize=(20, 10))
plt.subplot(141),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('unmasked k-space'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(ims, cmap = 'gray')
plt.title('masked k-space'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(img_back, cmap = 'gray')
plt.title('ifft'), plt.xticks([]), plt.yticks([])
plt.show()

fig, ax = plt.subplots(figsize=(20, 10))
plt.subplot(141),plt.imshow(img_rot,cmap='gray')
plt.title('rotation'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(magnitude_spectrum_90,cmap='gray')
plt.title('rotation 180'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(img_180,cmap='gray')
plt.title('rotation'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(magnitude_spectrum_180,cmap='gray')
plt.title('rotation 180'), plt.xticks([]), plt.yticks([])
plt.show()

img=np.array([[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]])
mask=np.array([[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0]])
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = (20*np.log(np.abs(fshift)))
s=fshift*(mask==1)
ims=20*np.log(np.abs(s))
f_ishift = np.fft.ifftshift(s)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

fig, ax = plt.subplots(figsize=(20, 10))
plt.subplot(141),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('unmasked k-space'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(ims, cmap = 'gray')
plt.title('masked k-space'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(img_back, cmap = 'gray')
plt.title('ifft'), plt.xticks([]), plt.yticks([])
plt.show()


img_rot=[[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]]
f = np.fft.fft2(img_rot)
fshift = np.fft.fftshift(f)
magnitude_spectrum = (20*np.log(np.abs(fshift)))
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.show()

img_rot=[[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]
iff=helper.iff(img_rot)
plt.imshow(iff, cmap = 'gray')
plt.show()
