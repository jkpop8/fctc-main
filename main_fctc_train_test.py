# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 00:36:27 2023

@author: jk
"""

import fctc.model as model
from sklearn.metrics import confusion_matrix
import os


#prepare dataset
from sklearn.datasets import load_iris
data = load_iris()
feat = data['data'] 
label = data['target']
#label = [0, 1, 2, 1, ..., N], -1 = no class

#prepare result file for model training
current_dir = os.path.dirname(os.path.abspath(__file__))
fn = os.path.join(current_dir, "fctc_model", "iris_all")
#fn = "fctc_model/iris_all"

fctc_model = model.Model()
#fit(train_feature, train_label, validation_feature, validation_label, result_filename, fold_no=0, norm=True)
fctc_model.fit(feat, label, feat, label, fn) 

#example of model prediction
predict_label, winners, confis = fctc_model.predict(feat) #predict(test_feature, norm=True)
confmat = confusion_matrix(label, predict_label)
print(confmat)


