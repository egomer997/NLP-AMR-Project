Contains the code for the project including exploration, cleaning, training classfication models, and the evaluation.

data_cretaion_and_eda used to check organsim distributions and filter data into pools to insert into chatgpt

Direct_models includes models that predict MRSA, CRE, VRE or NONE directly from text

Extract_and_rulebased includes models where organism, antibiotic and susceptibility are extracted before applying rule based classfication of MRSA, CRE, VRE and NONE

Joint_model compares models to predict bacteria, susceptibility, antibiotic and resistance group from the text directly as a joint model
it should be noted that the resistance group is dependent on the bacteria, susceptibility and antibiotic.

Functions:
Text Cleaning - clean
Train/Test Split - split_data
Model Training - train_and_test_model
Evaluation - evaluate_predictions

