# TFG-Machine-Learning-HEPhysics
The purpose of this repostery is to gather the code I have been developed lasts four months (as a final degree project in Physics, UB), for future students. If 
this aids to someone I would be really happy and proud. Here, it is studied two ML algorithms: Neural Netwoks (from PyTorch and TensorFlow) and the algorithm of Boosted Decision Trees on XGBoost's implementation.

This code contains two problems: whether the Lb0 decay to p,K,ee, from LHCb data, and one simulation of a banc problem (whether a person subscribes to a term deposit). The first one is the <strong>MAIN</strong> problem and the second one is only a personal trainment as a begginer in this gorgeous field of ML, were the data was extrated from a webpage mentioned in the README.

The aim is to study the models of PyTorch, TensorFlow and XGBoost (from Python) extracted of the documentation of each library and see how they perfom in each problem. Results and discusion and conclusion and beautiful-vectorial figures are on the pdf.

# Lb0 decay, from LHCb data
Lb0 ~ 5620 MeV of invariant mass.

This folder contains two files: Monte Carlo simulations of the decay, and real data of the LHCb data. The aim is to predict whether the decay happens in a rang of 4800 and 5800 MeV.

MonteCarlo: data to train the models the decay happens, converted as signal 1

LHCb data: above 5800 MeV data to train the models the decay does not happen, converted as 0

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
           around 4800 and 5800 MeV data to make the predictions
           
# Banc Deposit
Data extrated from https://www.section.io/engineering-education/machine-learning-with-xgboost-and-scikit-learn/, but in this repostery the focus is on each library, and all features of the datased of the reference is used. Furthermore, the file "modified" contains a crop of the datased of the reference for a 50%-50% of 1-0, for the feature "y", since NN models did not work properly on the real dataset otherwise.
