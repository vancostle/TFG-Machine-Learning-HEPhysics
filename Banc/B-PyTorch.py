#`3-Bank.Afiliation.csv` from https://www.section.io/engineering-education/machine-learning-with-xgboost-and-scikit-learn/
# https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1 This one use SK
# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/ This one only PD
# https://note.nkmk.me/en/python-pandas-len-shape-size/
#(https://pytorch.org/vision/0.8/_modules/torchvision/datasets/cifar.html#CIFAR10.__getitem__)
#Useful link: https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

import torch
torch.manual_seed(100)
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import pandas as pd
import numpy as np  

##################################################### Defining the class

class MyTorchDataset(Dataset):

  def __init__(self,file_name, train_size, train):
    df = pd.read_csv(file_name, sep=';')
    if (df.columns[df.dtypes == 'object']).any():
            df = pd.get_dummies(df, df.columns[df.dtypes == 'object'])

    size_train = int(df.shape[0]//(100/train_size))

    columns = len(df.columns)-1

    X_df_train = df.iloc[:size_train, :(columns-2)].values
    y_df_train = df.iloc[:size_train, columns].values
    X_df_test = df.iloc[size_train:, :(columns-2)].values
    y_df_test = df.iloc[size_train:, columns].values

    self.X_train = torch.tensor(X_df_train, dtype=torch.float32)
    self.y_train = torch.tensor(y_df_train, dtype=torch.long)
    self.X_test = torch.tensor(X_df_test, dtype=torch.float32)
    self.y_test = torch.tensor(y_df_test, dtype=torch.long)

    self.train = train

  def __len__(self):
    if self.train == 'train':
      return len(self.y_train)
    else:
      return len(self.y_test)
  
  def __getitem__(self,idx):
    if self.train == 'train':
      return self.X_train[idx], self.y_train[idx]
    else:
      return self.X_test[idx], self.y_test[idx]

datacsv = "3-bank-additional-modified.csv"

training_data = MyTorchDataset(datacsv, 80, 'train')
testing_data = MyTorchDataset(datacsv, 80, 'test')

###################################################### MODEL

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()                   # The super call delegates the function call to the parent class, nn.Module
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(            
            nn.Linear(61, 32),                                  # FIRST ARUGMENT NEED TO BE THE SAME LENGHT OF OUR DATA!!!!
            # nn.Linear(63, 32),                                # Bank-additional
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)                      # Why logits?
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()                                   
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)           
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()                                                                    # Telling to its model that have to be evaluated
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(testing_data, batch_size = batch_size)

for X, y in test_dataloader:
    print(f'Shape of X: {X.shape}; {X.dtype}')             # important to know the shape of our objects
    print(f'Shape of y: {y.shape}; {y.dtype}')             # in case to avoid model errors in the nn.Linear module 
    break

################################################# training
epochs = 30

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    
print("Done!")

############################################## A list/arrray is needed to make some plots, these are the following lines

predictions, actual, pred_prob, y_test= [],[],[],[]

def pred_func_PT(dataloader):
    model.eval()
    for i in range (0,((dataloader[:][0]).size()[0])):
        X_func_test = (dataloader[i][0])
        y_func_test = dataloader[i][1]
        y_test.append(y_func_test)
        with torch.no_grad():
            pred = nn.functional.softmax(model(X_func_test[None,:]),dim=1)
            pred_prob.append((1-pred[0][0]).item())
            predictions.append((pred[0].argmax(0)).item())
            actual.append(y_func_test.item())

pred_func_PT(testing_data)

############### Saving data and don't need to compile all this everytime you need to make a change in the figures

with open('PT-Bank-Actual-array.txt', 'w') as f:
    for item in actual:
        f.write("%s\n" % item)
f.close

with open('PT-Bank-Pred_prob-array.txt', 'w') as f:
    for item in pred_prob:
        f.write("%s\n" % item)
f.close
