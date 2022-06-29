import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from sklearn.metrics import roc_curve, auc

path = 'C:/Users/vanes/OneDrive - Universitat de Barcelona/8e semestre/TFG/TeX/Memoria/figures/'

def auc_roc_curve(actual, prob):
    fpr, tpr, threshold = roc_curve(actual, prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

############## ROC CURVES ####################################################
PT_actual = np.loadtxt("PT-Bank-Actual-array.txt", dtype=float)
PT_probab = np.loadtxt("PT-Bank-Pred_prob-array.txt", dtype=float)
TF_actual = np.loadtxt("TF-Bank-Actual-array.txt", dtype=float)
TF_probab = np.loadtxt("TF-Bank-Pred_prob-array.txt", dtype=float)
XGB_actual = np.loadtxt("XGB-Bank-Actual-array.txt", dtype=float)
XGB_probab = np.loadtxt("XGB-Bank-Pred_prob-array.txt", dtype=float)

PT_fpr, PT_tpr, PT_roc_auc = auc_roc_curve(PT_actual, PT_probab)
TF_fpr, TF_tpr, TF_roc_auc = auc_roc_curve(TF_actual, TF_probab)
XGB_fpr, XGB_tpr, XGB_roc_auc = auc_roc_curve(XGB_actual, XGB_probab)

fig, ax = plt.subplots(1,1,figsize=(3.4,2))
lw = 1
plt.plot(
    PT_fpr,
    PT_tpr, 
    lw=lw,
    label="PT-AUC = %0.2f" % PT_roc_auc,
    )

plt.plot(
    TF_fpr,
    TF_tpr, 
    lw=lw,
    label="TF-AUC = %0.2f" % TF_roc_auc,
    )

plt.plot(
    XGB_fpr,
    XGB_tpr, 
    lw=lw,
    label="XGB-AUC = %0.2f" % XGB_roc_auc,
    )

plt.tick_params(axis='both', direction="in", which='both', top=True, right=True)
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
ax.set_yticks([0.25,0.50,0.75,1.00])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

plt.savefig(path + 'Bank-ROC.pgf',bbox_inches='tight')
#############################################################################
