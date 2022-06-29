import pandas as pd                                         # loading and cleaning the dataset
import numpy as np                                          # manipulating it
import uproot as ut

import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(10)

#####################################################  Prepare the files

path_sig = r'C:\Users\vanes\OneDrive - Universitat de Barcelona\8e semestre\TFG\Code\CernData\MC.root'
path_bkg = r'C:\Users\vanes\OneDrive - Universitat de Barcelona\8e semestre\TFG\Code\CernData\data.root'

sig = ut.open(path_sig)['DecayTree;5']
bkg = ut.open(path_bkg)['DecayTree;1']

# Examined features, some of them need to be calculated
vact = ["beta", "Lb_PT", "Lambdastar_PT", "PT_LambdastarJspi", "Lb_IPCHI2_OWNPV", "Lambdastar_IPCHI2_OWNPV", "Lb_DIRA_OWNPV",
     "Lb_FDCHI2_OWNPV", "Jpsi_FDCHI2_OWNPV", "Lb_LOKI_DTF_CHI2NDOF", "Lb_ENDVERTEX_CHI2", "Lambdastar_ENDVERTEX_CHI2", 
     "PT_min_pK", "PT_pK", "IPCHI2_min_pK", "IPCHI2_pK", "PT_min_ll", "IPCHI2_min_ll", "IPCHI2_ll", "Proton_P", "ETA_pK"]
#############
vars = ["Lb_PT", "Lambdastar_PT", "Lb_IPCHI2_OWNPV", "Lambdastar_IPCHI2_OWNPV", "Lb_DIRA_OWNPV", "Lb_FDCHI2_OWNPV", 
"Jpsi_FDCHI2_OWNPV", "Lb_LOKI_DTF_CHI2NDOF", "Lb_ENDVERTEX_CHI2", "Lambdastar_ENDVERTEX_CHI2", "Proton_P", "Lb_M"]

# create a pandas data frame with these variables only
sig_df = sig.arrays(vars, library='pd')
bkg_df = bkg.arrays(vars, library='pd')
bkg_he = bkg.arrays(vars, library='pd')

tr = [sig, bkg, bkg]                 # trees
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

# now deleting the last column
del sig_df['Lb_M']
del bkg_df['Lb_M']
del bkg_he['Lb_M']

# converting to numpy array, needed to the model

sig_array = sig_df.to_numpy()
bkg_array = bkg_df.to_numpy()
bhe_array = bkg_he.to_numpy()
mas_test_array = mas_test.to_numpy()

print("Signal shape:", sig_array.shape)
print("Backgr shape:", bkg_array.shape)
print("Bck HE shape:", bhe_array.shape)

# merge and define signal and background labels
X_train = np.concatenate((sig_array[:1177], bhe_array))
X_test = bkg_array
y_train = np.concatenate((np.ones(sig_array[:1177].shape[0]),     # 1 is signal
                    np.zeros(bhe_array[:1177].shape[0])))  # 0 is background

print("X shape:", X_train.shape)
print("X shape:", X_test.shape)
print("y shape:", y_train.shape)

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)

print("Train size:", X_train.shape[0])
print("Test size: ", X_test.shape[0])

###################################################### MODEL

model = keras.Sequential([                                
    keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)), 
    keras.layers.BatchNormalization(),  
    keras.layers.Dense(100, activation='relu'), 
    keras.layers.Dense(2, activation='softmax')
])

# the last layer need to be 2, affiliation yes (1) or not (0), but with softmax we get their probabilities

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # optimizer='adam', # for predetermined parameters
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.2, epochs=500) # training the model


############################################## A list/arrray is needed to make some plots, these are the following lines for predictions

pred_prob = model.predict(X_test)
predictions, ye_clas, no_clas = []

for i in range (0,len(pred_prob)):
    predictions.append(np.argmax(pred_prob[i]))
    
actual = y_test.values.tolist()

for i in range (0,len(y_test)):
    if predictions[i] == 1:
        ye_clas.append(pred_prob[i,1])
    if predictions[i] == 0:
        no_clas.append(pred_prob[i,1])
    
############################################### Plots

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
fig, ax = plt.subplots(1,1,figsize=(3.4,2))
# plt.title('mod-data-XGB')
plt.minorticks_on()
ax.hist(no_clas, bins=10, facecolor='chocolate', edgecolor='#474746', alpha=1, label='background')
ax.hist(ye_clas, bins=10, facecolor='teal', edgecolor='#474746', alpha=1, label='signal')
ax.legend(loc='upper center',prop={'size':9},fancybox=True, framealpha=0.2)
plt.xlabel('Probability')
plt.xlim([-0,1])
plt.tight_layout()
plt.show()

plt.savefig(path + 'TF-LHCb-Histogram.pgf')

############### Saving data and don't need to compile all this everytime you need to make a change in the figures

with open('TF-LCHb-Actual-array.txt', 'w') as f:
    for i in range (0,len(actual)):
        f.write("%s\n" % actual[i][0])
f.close

with open('TF-LCHb-Pred_prob-array.txt', 'w') as f:
    for item in pred_prob[:,1]:
        f.write("%s\n" % item)
f.close

actual_f = np.loadtxt("TF-LCHb-Actual-array.txt", dtype=float)
pred_prob_f = np.loadtxt("TF-LCHb-Pred_prob-array.txt", dtype=float)


############### AUC-ROC curve

from sklearn.metrics import roc_curve, auc

# Compute micro-average ROC curve and ROC area
fpr, tpr, threshold = roc_curve(actual, pred_prob[:,1])
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
print(classification_report(y_test, predictions))

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
%matplotlib inline

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

cm = confusion_matrix(predictions, y_test)

fig, ax = plt.subplots(1,1,figsize=(2,1.5))
sns.heatmap(cm, annot = True)
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
plt.ylabel('Predicted class')
plt.xlabel('Actual class')

plt.savefig(path + 'TF-LHCb-Matrix.pgf',bbox_inches='tight')
