Instructions for running code

1.	Data_pooling_and_EDA should be ran on Data\Full raw data set\ Stanford_healthcare_data. Produced Data\Raw\(CRE_raw, MRSA_raw and VRE_raw), which are the files uploaded to LLM to produce synthetic text in Data\Synthetic.
2.	Direct_classification, extraction_then_rule_based and joint_model should be ran on Data\Synthetic(CRE_synthetic, MRSA_synthetic and VRE_synthetic) and the predictions are shown in the Data\(joined, extract or direct) file. Code\Functions is required for all code. 

**Final project plan**

This project uses Natural Language Processing (NLP) to extract clinically relevant information from messy microbiology culture reports using a synthetic dataset created from the Standford Healthcare data set. The aim was to classify antimicrobial resistance patterns.
Dataset: microbiology_cultures_cohort.csv 
Structured dataset from Stanford Healthcare:
The three features of interest were:
organism
antibiotic
susceptibility

The dataset was filtered into three datasets where the following 'positive' groups were the main focus:
Staphylococcus aureus, Resistant to Oxicillin → MRSA

Enterococcus faecalis/faecium Resistant to vancomycin → VRE

E. coli / Klebsiella pneumoniae Resistant to meropenem,
imipenem or orertapenem  → CRE

Created pools for each to include:
the positive cases above and negatives below:
A–E negative
A. Bacteria and drug of interest, but not resistant 
B. Other organisms sensitive to drug of interest
C. Other organisms resistant to drug of interest
D. Similar sounding bacteria
E. random negatives

Synthetic data was generated to produce messy text reports using LLM ChatGPT to Simulate real-world clinical text and evaluate how well machine learning models can:

Extract structured information (organism, antibiotic, susceptibility)
Classify resistance categories (MRSA, VRE, CRE, NONE)

The generated text includes:

abbreviations 
spelling variation
inconsistent formatting
clinical-style noise

This was to create realistic NLP notes that represent the healthcare challenges compared to clean structured data.


Three modelling approaches were implemented and throughout the models, both cleaned and raw data was used to compare how cleaning effects the models for the direct and extract approaches, however, just cleaned data was input into the joint models as it was estabilished that cleaning improved the performanace of the models. 

1. classification directly from messy data, where binary and multiclass models were explored:

Binary Classification Models

MRSA vs NOT_MRSA
VRE vs NOT_VRE
CRE vs NOT_CRE

Each model:

Input: clinical_report 
Output: binary label

Two versions were compared:

Raw text
Cleaned text (normalised using regex + standardisation rules)


Multiclass Classification Model

A single model was trained to predict:

MRSA
VRE
CRE
NONE

This combines all datasets into one classification task.

2. Extraction Models where structured data was extrcated including antibiotic, bacteria and sensitivity, and then a rule-based classification was applied to classify CRE, MRSA, VRE or NONE 

Separate models were trained to extract:

organism
antibiotic
susceptibility

From clinical report style data using supervised learning to simulate an information extraction pipeline:

clinical_report → predicted structured fields

Rule-Based Classifier - applied on extracted outputs to classify:

MRSA
VRE
CRE
NONE

Based on clinical logic:

organism + antibiotic + resistance → final label

This mimics real-world clinical decision rules.

3. Joint models
each model predicts organism, antibiotic, sensitivity and resistance group directly from cleaned text 

compares
logistic regression
support vector machine 
naive bayes 


Evaluation

Models were evaluated using:

Accuracy
Classification report
Confusion matrix
Error analysis

Both raw vs cleaned text performance was compared.

Outputs

The following result files were generated:

from pooling and eda:
MRSA_raw.csv
VRE_raw.csv
CRE_raw.csv

these were uploaded to chat gpt to create:
MRSA_synthetic.csv
VRE_synthetic.csv
CRE_synthetic.csv

which were input into the code to produce:

from direct:
Direct CRE_binary.csv
Direct MRSA_binary.csv
Direct VRE_binary.csv
Direct Ressitance_group_multinomial
containing organism, sensitivity and antibiotic and resistant group labels binary/multinomial

from extract:
extracted_and_rulebased_predictions,csv
contains organism, bacteria, sensitivity and label (vre,cre,mrsa or none) and then the predictions for cleaned and raw for bacteria, antibiotic and sensitivity, and then the rule based resistance label for cleaned and raw (mrsa,cre,vre or none)

from joint:
joined_model_predictions.csv

Summary

This project demonstrates how messy clinical text can be transformed into structured insights to reflect real-world challenges in microbiology and healthcare NLP systems.
