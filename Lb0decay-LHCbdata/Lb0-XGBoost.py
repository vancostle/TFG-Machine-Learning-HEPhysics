# Thanks to https://github.com/cmarinbe/lb2l1520mm_mariesamy/blob/master/train_bdt.py
# some dataframes and behaviour uproot4 https://github.com/scikit-hep/uproot4/discussions/114
# uproot3 to 4: https://uproot.readthedocs.io/en/latest/uproot3-to-4.html?highlight=Pandas#removed-features

import xgboost as xgb
import uproot as ut
import pandas as pd
import numpy as np      # manipulating it

import sklearn
from sklearn.model_selection import train_test_split    # split data intro train and test

#####################################################  Prepare the files

# read the data
sig = ut.open(r'C:\Users\vanes\OneDrive - Universitat de Barcelona\8e semestre\TFG\Code\CernData\MC.root')['DecayTree;5']
bkg = ut.open(r'C:\Users\vanes\OneDrive - Universitat de Barcelona\8e semestre\TFG\Code\CernData\data.root')['DecayTree;1']

# bkg.show()
# prove = sig.arrays("Lb_PT", entry_stop=stop1, library='pd')["Lb_PT"] + sig.arrays("Lambdastar_PT", entry_stop=entrystop, library='pd')["Lambdastar_PT"]

totalvars_sig = sig.keys()
totalvars_bkg = bkg.keys()

vact = ["beta", "Lb_PT", "Lambdastar_PT", "PT_LambdastarJspi", "Lb_IPCHI2_OWNPV", "Lambdastar_IPCHI2_OWNPV", "Lb_DIRA_OWNPV",
     "Lb_FDCHI2_OWNPV", "Jpsi_FDCHI2_OWNPV", "Lb_LOKI_DTF_CHI2NDOF", "Lb_ENDVERTEX_CHI2", "Lambdastar_ENDVERTEX_CHI2", 
     "PT_min_pK", "PT_pK", "IPCHI2_min_pK", "IPCHI2_pK", "PT_min_ll", "IPCHI2_min_ll", "IPCHI2_ll", "Proton_P", "ETA_pK"]

vars = ["Lb_PT", "Lambdastar_PT", "Lb_IPCHI2_OWNPV", "Lambdastar_IPCHI2_OWNPV", "Lb_DIRA_OWNPV", "Lb_FDCHI2_OWNPV", 
    "Jpsi_FDCHI2_OWNPV", "Lb_LOKI_DTF_CHI2NDOF", "Lb_ENDVERTEX_CHI2", "Lambdastar_ENDVERTEX_CHI2", "Proton_P", "Lb_M"]

# create a pandas data frame with these variables only
sig_df = sig.arrays(vars, library='pd')        # will be used as 1 for train
bkg_df = bkg.arrays(vars, library='pd')        # will be used as test
bkg_he = bkg.arrays(vars, library='pd')        # will be used as 0 for train


tr = [sig, bkg, bkg]                    # trees
df = [sig_df, bkg_df, bkg_he]           # dataframes

# Calculating beta variable

def insert_beta(tr, df):
    for i in range (0,3):
        pJpsi = tr[i].arrays('Jpsi_P', library='pd')['Jpsi_P']
        pp = tr[i].arrays('Proton_P', library='pd')['Proton_P']
        pK = tr[i].arrays('Kaon_PT', library='pd')['Kaon_PT']

        num = pJpsi - pp - pK
        den = pJpsi + pp + pK

        df[i].insert(0, 'beta', num / den + 1)

# creating the new variables and adding to a column
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

# filtering the values of the mass
bkg_df = bkg_df[bkg_df['Lb_M'] < 5800]
bkg_he = bkg_he[bkg_he['Lb_M'] > 5800]

# deleting the negatives values on IPCHI2
bkg_df = bkg_df[bkg_df["IPCHI2_pK"] > 0]
bkg_he = bkg_he[bkg_he["IPCHI2_pK"] > 0]
bkg_df = bkg_df[bkg_df["IPCHI2_min_pK"] > 0]
bkg_he = bkg_he[bkg_he["IPCHI2_min_pK"] > 0]

# now deleting the last column
del sig_df['Lb_M']
del bkg_df['Lb_M']
del bkg_he['Lb_M']

#converting to numpy array

sig_array = sig_df.to_numpy()
bkg_array = bkg_df.to_numpy()
bhe_array = bkg_he.to_numpy()

print("Signal shape:", sig_array.shape)
print("Backgr shape:", bkg_array.shape)
print("Bck HE shape:", bhe_array.shape)

# merge and define signal and background labels
X = np.concatenate((sig_array[:1177], bhe_array))
y = np.concatenate((np.ones(sig_array[:1177].shape[0]),     # 1 is signal
                    np.zeros(bhe_array[:1177].shape[0])))   # 0 is background

print("X shape:", X.shape)
print("y shape:", y.shape)

# split data in train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
print("Train size:", X_train.shape[0])
print("Test size: ", X_test.shape[0])

############################################# Training the model, this is very easy

xgb_classifier = xgb.XGBClassifier() 
xgb_classifier.fit(X_train,y_train)  

############################################# Predictions

predictions = xgb_classifier.predict(X_test)              # making predictions
pred_prob = xgb_classifier.predict_proba(X_test)          # extrating the probability of the predictions

############################################# Accuracy of the model
from sklearn.metrics import accuracy_score

print("Accuracy of Model::",accuracy_score(y_test,predictions))  # accuracy


############################################## A list/arrray is needed to make some plots, these are the following lines

actual = pd.Series(y_test).values

no_clas = []
ye_clas = []

for i in range (0,len(predictions)):
    if predictions[i] == 1:
        ye_clas.append(pred_prob[i,1])
    if predictions[i] == 0:
        no_clas.append(pred_prob[i,1])

############################################### Plotting things

path='C:/Users/vanes/OneDrive - Universitat de Barcelona/8e semestre/TFG/TeX/Memoria/figures/'

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
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
# plt.title('mod-data-XGB')
plt.minorticks_on()
ax.hist(no_clas, bins=10, facecolor='chocolate', edgecolor='#474746', alpha=1, label='background')
ax.hist(ye_clas, bins=10, facecolor='teal', edgecolor='#474746', alpha=1, label='signal')
ax.legend(loc='upper center',prop={'size':9},fancybox=True, framealpha=0.2)
plt.xlabel('Probability')
plt.xlim([-0,1])
plt.tight_layout()
plt.show()

plt.savefig(path + 'XGB-LHCb-Histogram.pgf',bbox_inches='tight')

############### Saving data and don't need to compile all this everytime you need to make a change in the figures

with open('XGB-LCHb-Actual-array.txt', 'w') as f:
    for item in actual:
        f.write("%s\n" % item)
f.close

with open('XGB-LCHb-Pred_prob-array.txt', 'w') as f:
    for item in pred_prob[:,1]:
        f.write("%s\n" % item)
f.close

actual_f = np.loadtxt("XGB-LCHb-Actual-array.txt", dtype=float)
pred_prob_f = np.loadtxt("XGB-LCHb-Pred_prob-array.txt", dtype=float)

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
     label="ROC curve (area = %0.3f)" % roc_auc,
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

plt.style.use('default')
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(predictions, y_test)

fig, ax = plt.subplots(1,1,figsize=(2,1.5))
res=sns.heatmap(cm, annot = True, cbar=False)
for _, spine in res.spines.items():
    spine.set_visible(True)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none') 
fig.colorbar(ax.get_children()[0])
plt.ylabel('Predicted class')
plt.xlabel('Actual class')
plt.show()

plt.savefig(path + 'XGB-LHCb-Matrix-spine.pgf',bbox_inches='tight')
