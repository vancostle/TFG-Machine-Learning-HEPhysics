import torch
import uproot as ut
import pandas as pd
import numpy as np  

torch.manual_seed(1000)                               # https://pytorch.org/docs/stable/notes/randomness.html
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import sklearn
from sklearn.model_selection import train_test_split 

#####################################################  Prepare the files

path_sig = r'C:\Users\vanes\OneDrive - Universitat de Barcelona\8e semestre\TFG\Code\CernData\MC.root'
path_bkg = r'C:\Users\vanes\OneDrive - Universitat de Barcelona\8e semestre\TFG\Code\CernData\data.root'

sig = ut.open(path_sig)['DecayTree;5']
bkg = ut.open(path_bkg)['DecayTree;1']

# Examined features, some of them need to be calculated

# vact = ["beta", "Lb_PT", "Lambdastar_PT", "PT_LambdastarJspi", "Lb_IPCHI2_OWNPV", "Lambdastar_IPCHI2_OWNPV", "Lb_DIRA_OWNPV",
#     "Lb_FDCHI2_OWNPV", "Jpsi_FDCHI2_OWNPV", "Lb_LOKI_DTF_CHI2NDOF", "Lb_ENDVERTEX_CHI2", "Lambdastar_ENDVERTEX_CHI2", 
#     "PT_min_pK", "PT_pK", "IPCHI2_min_pK", "IPCHI2_pK", "PT_min_ll", "IPCHI2_min_ll", "IPCHI2_ll", "Proton_P", "ETA_pK"]

vars = ["Lb_PT", "Lambdastar_PT", "Lb_IPCHI2_OWNPV", "Lambdastar_IPCHI2_OWNPV", "Lb_DIRA_OWNPV", "Lb_FDCHI2_OWNPV", 
"Jpsi_FDCHI2_OWNPV", "Lb_LOKI_DTF_CHI2NDOF", "Lb_ENDVERTEX_CHI2", "Lambdastar_ENDVERTEX_CHI2", "Proton_P", "Lb_M"]

# create a pandas data frame with these variables only
sig_df = sig.arrays(vars, library='pd')
bkg_df = bkg.arrays(vars, library='pd')
bkg_he = bkg.arrays(vars, library='pd')

tr = [sig, bkg, bkg]                    # trees
df = [sig_df, bkg_df, bkg_he]           # dataframes

def prepare_root(tr, df):
    # Calculating beta variable
    def insert_beta(tr, df):
        for i in range (0,3):
            pJpsi = tr[i].arrays('Jpsi_P', library='pd')['Jpsi_P']
            pp = tr[i].arrays('Proton_P', library='pd')['Proton_P']
            pK = tr[i].arrays('Kaon_PT', library='pd')['Kaon_PT']

            num = pJpsi - pp - pK
            den = pJpsi + pp + pK

            df[i].insert(0, 'beta', num / den + 1)

    # Creating the new variables and adding to a column
    def insert_column(tr, df, index, column_name1, column_name2, beg_, sum, min, part):
        # first two arg:    total
        # second two ar:    partial df with the scrutinized vars
        # index:            where to place it on the df 
        # columns_name$:    columns to operate
        # beg_:             PT_, IP_ or ETA_
        # part:             particles involved

        for i in range (0,3):
            col1 = tr[i].arrays(column_name1, library='pd')[column_name1]
            col2 = tr[i].arrays(column_name2, library='pd')[column_name2]

            sum_df = col1 + col2
            min_df = pd.concat([col1,col2]).groupby(level=0).min()
            
            if min == True:
                df[i].insert( index , beg_ + "min_" + part, min_df)

            if sum == True:
                df[i].insert( index + 1 , beg_ + part, sum_df)
    
    insert_beta(tr, df)
    insert_column(tr, df, 2, 'Lambdastar_PT', 'Jpsi_PT', 'PT_', True, False, 'LambdastarJpsi')
    insert_column(tr, df, 12, 'Proton_PT', 'Kaon_PT', 'PT_', True, True, 'pK')
    insert_column(tr, df, 14, 'Proton_IPCHI2_ORIVX', 'Kaon_IPCHI2_ORIVX', 'IPCHI2_', True, True, 'pK')
    insert_column(tr, df, 16, 'L1_PT', 'L2_PT', 'PT_', False, True, 'll')
    insert_column(tr, df, 17, 'L1_IPCHI2_ORIVX', 'L2_IPCHI2_ORIVX', 'IPCHI2_', True, True, 'll')
    insert_column(tr, df, 19, 'Proton_ETA', 'Kaon_ETA', 'ETA_', True, False, 'pK')

    return df[0], df[1]

sig_df, bkg_df = prepare_root(tr, df)

# filtering the values of the mass
bkg_df = bkg_df[bkg_df['Lb_M'] < 5800]
bkg_he = bkg_he[bkg_he['Lb_M'] > 5800]

# deleting the negatives values on IPCHI2
bkg_df = bkg_df[bkg_df["IPCHI2_pK"] > 0]
bkg_he = bkg_he[bkg_he["IPCHI2_pK"] > 0]
bkg_df = bkg_df[bkg_df["IPCHI2_min_pK"] > 0]
bkg_he = bkg_he[bkg_he["IPCHI2_min_pK"] > 0]

# saving the mass for prediction plot
mas_test = bkg_df[['Lb_M']]
mas_test_array = mas_test.to_numpy()

# now deleting the last column
del sig_df['Lb_M']
del bkg_df['Lb_M']
del bkg_he['Lb_M']

##################################################### Defining the class

class MyTorchDataset(Dataset):   # PyTorch is needed to define a class for the model used

  def __init__(self, sig_df, bkg_he, train_size, train):
    sig_array = sig_df.to_numpy()
    bhe_array = bkg_he.to_numpy()

    X = np.concatenate((sig_array[:1177], bhe_array[:1177]))
    y = np.concatenate((np.ones(sig_array[:1177].shape[0]),     # 1 is signal
                        np.zeros(bhe_array[:1177].shape[0])))   # 0 is background

    # split data in train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_size)
    self.X_train = torch.from_numpy(X_train).clone().float()
    self.y_train = torch.from_numpy(y_train).clone().long()
    self.X_test = torch.from_numpy(X_test).clone().float()
    self.y_test = torch.from_numpy(y_test).clone().long()

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
    
###################################################### Data is well prepared now :)

training_data = MyTorchDataset(sig_df, bkg_he, 0.5, 'train')
testing_data = MyTorchDataset(sig_df, bkg_df, 0.5, 'test')

###################################################### MODEL

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()                   
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(            
            nn.Linear(21, 32),                                  # FIRST ARUGMENT NEED TO BE THE SAME LENGHT OF OUR DATA!!!!
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)                                    # if second argument =19 the model dont give an error
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)                      
        return logits

model = NeuralNetwork().to(device)

print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
    model.eval()                                                                    
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

# Create data loaders, futher information: https://blog.paperspace.com/dataloaders-abstractions-pytorch/
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(testing_data, batch_size = batch_size)

for X, y in test_dataloader:
    print(f'Shape of X: {X.shape}; {X.dtype}')             # important to know the shape of our objects
    print(f'Shape of y: {y.shape}; {y.dtype}')             # in case to avoid model errors in the nn.Linear module 
    break
 
############################################## Let's train :) !!

epochs = 500

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    
print("Done!")

############################################## A list/arrray is needed to make some plots, these are the following lines

predictions, actual, pred_prob, y_test = [], [], [], []
no_clas, ye_clas = [], []

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
            if (i % 300) == 0:
                print(f'iteration: "{i}", test row: "{X_func_test[0]}", prob: "{pred[0][0]}", \
prediction: "{pred[0].argmax(0)}", actual value: "{y_func_test}"')

pred_func_PT(testing_data)

for i in range (0,len(y_test)):
    if predictions[i] == 1:
        ye_clas.append(pred_prob[i])
    if predictions[i] == 0:
        no_clas.append(pred_prob[i])
        
############################################### Plotting things

path='C:/Users/vanes/OneDrive - Universitat de Barcelona/8e semestre/TFG/TeX/Memoria/figures/'

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.style.use('classic')
%matplotlib inline

#######
# VERY IMPORTANT LINES FOR FANCY LATEX PLOTS <3 But you have to know that "para presumir hay que sufrir" 
# and this lines do not like the command "plt.show()"
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
#######

####### Distribuion of the probability of signal

fig, ax = plt.subplots(1,1,figsize=(3.4,2))
plt.minorticks_on()
ax.hist(no_clas, bins=10, facecolor='chocolate', edgecolor='#474746', alpha=1, label='background')
ax.hist(ye_clas, bins=10, facecolor='teal', edgecolor='#474746', alpha=1, label='signal')
ax.legend(loc='upper center',prop={'size':9},fancybox=True, framealpha=0.2)
plt.xlabel('Probability')
plt.xlim([0,1])
plt.tight_layout()
plt.show()

plt.savefig(path + 'PT-LHCb-Histogram.pgf')

############### Saving data and don't need to compile all this everytime you need to make a change in the figures

with open('PT-LCHb-Actual-array.txt', 'w') as f:
    for item in actual:
        f.write("%s\n" % item)
f.close

with open('PT-LCHb-Pred_prob-array.txt', 'w') as f:
    for item in pred_prob:
        f.write("%s\n" % item)
f.close

actual_f = np.loadtxt("PT-LCHb-Actual-array.txt", dtype=float)
pred_prob_f = np.loadtxt("PT-LCHb-Pred_prob-array.txt", dtype=float)

############### AUC-ROC curve

from sklearn.metrics import roc_curve, auc

# Compute micro-average ROC curve and ROC area
fpr, tpr, threshold = roc_curve(actual_f, pred_prob_f)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()

################## Confusion Matrix

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(actual, predictions))

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
%matplotlib inline

cm = confusion_matrix(predictions, actual)

fig, ax = plt.subplots(1,1,figsize=(2,1.5))
sns.heatmap(cm, annot = True)
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
plt.ylabel('Predicted class')
plt.xlabel('Actual class')

plt.savefig(path + 'PT-LHCb-Matrix.pgf',bbox_inches='tight')

############ Predinting the decay on a 4800 and 5800 MeV

mas_pred_1, mas_pred_2, mas_test = [], [], mas_test_array[:,0].tolist()

plt.style.use('classic')
%matplotlib inline

for i in range(0,len(predictions)):
    if pred_prob[i] >= 0.98:
        mas_pred_1.append(mas_test[i])
    if pred_prob[i] >= 0.99:          # mass_2 is only a larger threshold
        mas_pred_2.append(mas_test[i])

fig, ax = plt.subplots(1,1,figsize=(2.45,2.1))
# fig, ax = plt.subplots(1,1,figsize=(3.4,2))
plt.title('PyTorch')
plt.minorticks_on()
ax.hist(mas_pred_1, bins=70, facecolor='lightgray', edgecolor='#474746', histtype='stepfilled', align='mid', alpha=1, label='0.98')
# ax.hist(mas_pred_2, bins=70, facecolor='lightgray', edgecolor='#474746', align='mid', alpha=1, label='0.99')       
ax.legend(loc='upper left',prop={'size':9},fancybox=True, framealpha=0.2)               
plt.xlim([4900, 5800])    
ax.set_xticks([5000,5300,5600])
plt.ylim([0, 13])          
# ax.set_yticks([0,5,10])
plt.xlabel('Mass (MeV)')
plt.tight_layout()
plt.show()

path = 'C:/Users/vanes/OneDrive - Universitat de Barcelona/8e semestre/TFG/TeX/Memoria/figures/'
plt.savefig(path+'PT-LambdaMass.pgf',bbox_inches='tight')

# That's all folks
