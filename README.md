# fctc menuscript
Fuzzy Clustering Tree Classifier
https://ieeexplore.ieee.org/document/9684646

# manual
<table>
<tr><th>1. main_fctc_train_test.py</th><th>an example of train and validate in 1-fold crossvalidation</th></tr>
<tr><td>fctc_model.fit()</td><td>train</td></tr>
<tr><td>label_names, winner_indexes, confidences = fctc_model.predict()</td><td>validate</td></tr>
<tr><td></td><td>output files are created in fctc_model folder consisting of</td></tr>
<tr><td>*_result.txt</td><td>prediction results</td></tr>
<tr><td></td><td>acc = max(h_ac_valid)</td></tr>
<tr><td></td><td>f1 = max(h_f1_valid)</td></tr>
<tr><td>*_rule3.txt</td><td>if-then extracted rules</td></tr>
<tr><td>*_model_la.csv</td><td>labels of prototypes in the trained model</td></tr>
<tr><td>*_model_za.csv</td><td>features of prototypes in the trained model</td></tr>
<tr><th>2. main_fctc_load_test.py</th><th>an example of load and test the trained model</th></tr>
<tr><td>fctc_model.load()</td><td>load the model from fctc_model folder</td></tr>
<tr><td>label_names, winner_indexes, confidences = fctc_model.predict()</td><td>test</td></tr>
</table>
