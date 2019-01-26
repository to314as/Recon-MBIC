import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import datasets, transforms
import helper
from IPython.display import clear_output


nb_samples = 100
batch_size = 16
time_max_rim = 5
time_max_gd = 60
nb_epochs = 2
sigma_noise = 0.2
mask=cv2.imread('C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\mask\\cartes\\cartes_50.tif',0)


size_train = 100
size_test = 100

def loss(x, y):
    return 0.5 * x.T.dot(x) -  y.T.dot(x)

def gradient(x, y):
    return x - y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
transform = transforms.Compose([transforms.Resize(256),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])
batch_size = 20
# load the training and test datasets
train_data = datasets.ImageFolder(root='C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\brain\\db_train',
                                    transform=transform)
X_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size)#.numpy()[:size_train].reshape(size_train, 784) / 255
Y_train = helper.undersample(X_train, mask)
X0_train = Y_train

test_data = datasets.ImageFolder(root='C:\\Users\\tobi9\\Documents\\GitHub\\RefineGAN\\data\\brain\\db_valid',
                                   transform=transform)
X_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size)#.numpy()[:size_test].reshape(size_test, 784) / 255
Y_test = helper.undersample(X_test, mask)
X0_test = Y_test

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, init_variance=0.1):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.h1_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2_layer = nn.Linear(hidden_size*2, hidden_size)
        self.h3_layer = nn.Linear(hidden_size*2, hidden_size)
        self.output_layer = nn.Linear(hidden_size*2, output_size)
        
        nn.init.xavier_normal(self.h1_layer.weight, init_variance)
        nn.init.xavier_normal(self.h2_layer.weight, init_variance)
        nn.init.xavier_normal(self.h3_layer.weight, init_variance)
        nn.init.xavier_normal(self.output_layer.weight, init_variance)
        
    def forward(self, xt, st):
        combined_h1 = torch.cat((xt, st[0]), 2)
        h1 = self.h1_layer(combined_h1)
        h1 = f.relu(h1)
        
        combined_h2 = torch.cat((h1,st[1]), 2)
        h2 = self.h2_layer(combined_h2)
        h2 = f.relu(h2)
        
        combined_h3 = torch.cat((h2,st[2]), 2)
        h3 = self.h3_layer(combined_h3)
        h3 = f.relu(h3)
        
        output = self.output_layer(combined_h3)
        
        return output, [h1, h2, h3]
    
def weight_loss(t, T):
#     return t/T
    return 1

def psi(h):
    x = f.relu(h)
    x = 1-f.relu(-x+1)
    return x

class RIM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_max):
        super(RIM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.time_max = time_max
        self.rnn = RNN(input_size, hidden_size, output_size)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, ht, st, y):
        input = Variable(torch.Tensor(np.concatenate([ht.data.numpy(), 0*gradient(ht.data.numpy(), y)], 0).reshape(1, 1, self.input_size)))
        out, st = self.rnn.forward(input, st)
        return out.view(self.output_size), st
    
    def backprop(self, loss): 
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def forward_sequence(self, x0, y, x_true, phase="test"): 
        ht = Variable(torch.Tensor(x0)) # xt = \phi(ht) => see paper, where they use \eta instead of h
        st = [self.init_hidden(), self.init_hidden(), self.init_hidden()]
        
        loss = 0
        list_xt = [x0]
        for t in range(self.time_max):
            dh, st = self.forward(ht, st, y)
            ht = ht + dh
            xt = psi(ht)
            loss += weight_loss(t,self.time_max) * self.criterion(xt, Variable(torch.Tensor(x_true)))
#             print("loss in loop", loss)
            list_xt.append(xt.data.numpy())
#         print("loss forward sequence", loss)
            
        list_xt = np.array(list_xt)
        loss = loss / self.time_max
        
        if phase == "train":
            self.backprop(loss)
        
        return list_xt, loss
        
    def train_batch(self, X_batch, Y_batch, X0_batch):
        batch_loss = 0
        for i in range(len(X_batch)):
            x, y, x0 = X_batch[i], Y_batch[i], X0_batch[i]
            
            list_xt, loss = self.forward_sequence(x0, y, x, "test")
#             print("loss train batch", loss)
            batch_loss += loss
        
        batch_loss /= len(X_batch)
        self.backprop(batch_loss)
        
        return batch_loss.data.numpy()[0]
    
    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))
    
rim = RIM(X_train.shape[1]*2, X_train.shape[1], X_train.shape[1], time_max_rim)
list_losses = []
# rim.optimizer.param_groups[0]["lr"] /= 10
    
nb_epochs = 20
for epoch in range(nb_epochs):
    X_train, Y_train, X0_train = np.random.shuffle(X_train, Y_train, X0_train)
    
    loss_epoch = 0
    for i_begin_batch in tqdm(range(0, size_train, batch_size)):
        i_end_batch = (i_begin_batch+batch_size)
        if i_end_batch > size_train:
            next
        X_batch, Y_batch = X_train[i_begin_batch:i_end_batch], Y_train[i_begin_batch:i_end_batch]
        X0_batch = X0_train[i_begin_batch:i_end_batch]
        loss_batch = rim.train_batch(X_batch, Y_batch, X0_batch)
        loss_epoch += loss_batch
        
    loss_epoch /= (size_train // batch_size)
    list_losses.append(loss_epoch)
    
    clear_output(wait=True)
    print("Epoch {0}, loss: {1:05f}".format(epoch,loss_epoch))

plt.plot(list_losses)


list_xt_rim, losses = [], []
for (x0, y, x) in zip(X0_test, Y_test, X_test):
    list_xt, los = rim.forward_sequence(x0, y, x, "test")
    list_xt_rim.append(list_xt)
    losses.append(los.data.numpy()[0])
list_xt_rim = np.array(list_xt_rim)
losses = np.array(losses)

i_example = 48
t_final = 4

print("Loss: {0:5f}".format(float(losses[i_example])))

plt.rcParams['figure.figsize'] = (20,10)

for i in range(1,6):
    plt.subplot(1,6,i)
    plt.imshow(list_xt_rim[i_example][i-1].reshape(28,28), vmin=0, vmax=1)
    plt.title("t={}".format(i-1))
plt.subplot(1,6,6)
plt.imshow(X_test[i_example].reshape(28,28))
plt.title("Truth")