#%%
import os
import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, image 

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class CatsAndDogsDataset(Dataset):
    """Cats and Dogs dataset."""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.cat_dir = os.path.join(self.root_dir, 'cats')
        self.dog_dir = os.path.join(self.root_dir, 'dogs')
        self.transform = transform
        self.sample_list = []
        with os.scandir('../data/cats_vs_dog_small/train/') as it:
            label = 0
            for entry in it:
                if(entry.is_dir()):
                    with os.scandir(entry.path) as it:
                        for entry in it:
                            self.sample_list.append({"image":entry.path, "label":label }) 
                    label = label + 1
        random.shuffle(self.sample_list)
        #print(self.sample_list)
        self.sample_size = len(self.sample_list)

    def __len__(self):
        return(self.sample_size)

    def __getitem__(self, idx):
        if idx >= self.sample_size:
            print('ERROR: index out of range')
        img_name = self.sample_list[idx]['image']
        label = self.sample_list[idx]['label']
        img = read_image(img_name, image.ImageReadMode.RGB)
        #img = img.to(torch.float)
        #img = T.Resize((150,150))(img)
        img = T.functional.to_pil_image(img)

        if self.transform:
            img = self.transform(img)

        #return img, torch.as_tensor(label,dtype=torch.float32)
        return img, label
 
      
class CatsDogs(nn.Module):
    def __init__(self):
        super(CatsDogs, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3) 
        self.conv2 = nn.Conv2d(32,64,3) 
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,128,3)
        self.activation1 = nn.ReLU()
        self.maxpool2d1 = nn.MaxPool2d(2)
        self.maxpool2d2 = nn.MaxPool2d(2)
        self.maxpool2d3 = nn.MaxPool2d(2)
        self.maxpool2d4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(6272,512) # 128x7x7
        self.fc2 = nn.Linear(512,1)
        self.fltn = nn.Flatten()
        self.activation2 = nn.Sigmoid()

    def forward(self,x):
        # Image Size 150x150
        # x = F.max_pool2d(F.relu(self.conv1(x)),[2,2]) # 148x148 => 74x74
        # x = F.max_pool2d(F.relu(self.conv2(x)),[2,2]) #  72x72 => 36x36
        # x = F.max_pool2d(F.relu(self.conv3(x)),2) #  34x34 => 17x17
        # x = F.max_pool2d(F.relu(self.conv4(x)),2) #  15x15 => 7x7
        # x = self.fltn(x)
        # x = F.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))

        # Layer 1
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.maxpool2d1(x)
        
        # Layer 2 
        x = self.conv2(x)
        x = self.activation1(x)
        x = self.maxpool2d2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.activation1(x)
        x = self.maxpool2d3(x)

        # Layer 4
        x = self.conv4(x)
        x = self.activation1(x)
        x = self.maxpool2d4(x)

        x = self.fltn(x)

        # Layer 5
        x = self.fc1(x)
        x = self.activation1(x)

        # Layer 6
        x = self.fc2(x)
        x = self.activation2(x)

        return x


batch_size = 20

train_dataset = CatsAndDogsDataset('../data/cats_vs_dog_small/train', transform=T.Compose([
                                               T.Resize([150,150]),
                                               T.ToTensor()
                                           ]))

test_dataset = CatsAndDogsDataset('../data/cats_vs_dog_small/test', transform=T.Compose([
                                               T.Resize([150,150]),
                                               T.ToTensor()
                                           ]))

val_dataset = CatsAndDogsDataset('../data/cats_vs_dog_small/validation', transform=T.Compose([
                                               T.Resize([150,150]),
                                               T.ToTensor()
                                           ]))

# train_dataset = datasets.ImageFolder('../data/cats_vs_dog_small/train', transform=T.Compose([
#                                                T.Resize([150,150]),
#                                                T.ToTensor(),
#                                                T.Normalize([0.5]*3, [0.5]*3)
#                                            ]))
# test_dataset = datasets.ImageFolder('../data/cats_vs_dog_small/test', transform=T.Compose([
#                                                T.Resize([150,150]),
#                                                T.ToTensor(),
#                                                T.Normalize([0.5]*3, [0.5]*3)
#                                            ]))
# val_dataset = datasets.ImageFolder('../data/cats_vs_dog_small/validation', transform=T.Compose([
#                                                T.Resize([150,150]),
#                                                T.ToTensor(),
#                                                T.Normalize([0.5]*3, [0.5]*3)
#                                            ]))

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

##Test Dataloader
# for X, y in train_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape} {X.dtype}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     #print(X)
#     print(y)
#     plt.imshow(T.functional.to_pil_image(X[1]))
#     break

model = CatsDogs().to(device)
print("Model \n -------------------------------")
print(model)
print("-------------------------------")

#Test your model
# input = torch.rand(1, 3, 150, 150).to(device)
# print('\nImage batch shape:')
# print(input.shape)


# output = model(input) 
# print('\nRaw output:')
# print(output[0]) 
# print(f"Shape of y: {output.shape} {output.dtype}")

loss_fn = nn.BCELoss()
#loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    correct, avg_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.float().unsqueeze(1).to(device)
        
        # Compute prediction error
        pred = model(X)
        #print(pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += ((pred >= 0.5).type(torch.float) == y).type(torch.float).sum().item()
        avg_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss.append(avg_loss/num_batches)
    train_acc.append(correct/size)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.float().unsqueeze(1).to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += ((pred >= 0.5).type(torch.float) == y).type(torch.float).sum().item()

    avg_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    val_acc.append(correct)
    val_loss.append(avg_loss)
    print(avg_loss)

train_loss =[]
train_acc = []
val_acc =[]
val_loss = []

def main():
    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_acc, 'bo', label='Training acc') 
    plt.plot(epochs, val_acc, 'b', label='Test acc') 
    plt.title('Training accuraccy and test accuracy') 
    plt.legend()
    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='Training loss') 
    plt.plot(epochs, val_loss, 'b', label='Test loss') 
    plt.title('Training loss and test loss') 
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    print("Done!")

if __name__ == '__main__':
    main()






# %%
