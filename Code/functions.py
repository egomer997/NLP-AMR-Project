#imports
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

#text cleaning function
def clean(text: str) -> str:

    #handles missing NaN values
    if pd.isna(text):
        return ""

    #converts text to lowercase for consistancy
    text = str(text).lower()

    # remove unwanted punctuation, keepa / . - as they're used clinically
    text = re.sub(r"[^a-z0-9\s./-]", " ", text)

    # normalise spaces, multiple into single spaces
    text = re.sub(r"\s+", " ", text).strip()

    #these standardisations,
    #are standardising common short-hand abbreviations into standard form
    #this section could be developed further to include more well known terms

    # standardise susceptibility terms
    text = re.sub(r"\br\b", "resistant", text)
    text = re.sub(r"\bres\b", "resistant", text)
    text = re.sub(r"\bs\b", "susceptible", text)
    text = re.sub(r"\bsensitive\b", "susceptible", text)
    text = re.sub(r"\bi\b", "intermediate", text)

    # antibiotic shorthand
    text = re.sub(r"\bvanc\b", "vancomycin", text)
    text = re.sub(r"\bcipro\b", "ciprofloxacin", text)
    text = re.sub(r"\bfox\b", "cefoxitin", text)
    text = re.sub(r"\bcfox\b", "cefoxitin", text)

    # organism variations
    text = re.sub(r"\be\.?\s?coli\b", "escherichia coli", text)
    text = re.sub(r"\bk\.?\s?pneumoniae\b", "klebsiella pneumoniae", text)
    text = re.sub(r"\bkleb pneumo\b", "klebsiella pneumoniae", text)
    text = re.sub(r"\bs\.?\s?aureus\b", "staphylococcus aureus", text)
    text = re.sub(r"\bstaph aureus\b", "staphylococcus aureus", text)
    text = re.sub(r"\be\.?\s?faecalis\b", "enterococcus faecalis", text)
    text = re.sub(r"\be\.?\s?faecium\b", "enterococcus faecium", text)

    #clean up of spacing again after making changes to text
    text = re.sub(r"\s+", " ", text).strip()
    return text

#test train split function
#80% testing, 20% training
#stratified used to maintain class distribution
#fixed the random state for reproducibility throughout the project
def split_data(df, label_col="label"):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[label_col]
    )
    return train_df, test_df

#train and test function
#trains the model to make predictions on test data
# using TF-IDF and logistic regression
def train_and_test_model(train_df, test_df, text_column, target_column):

    #extract training features and labels
    X_train = train_df[text_column]
    y_train = train_df[target_column]

    #extract test features
    X_test = test_df[text_column]

    #creates a pipeline where text is turned into
    # numerical features to input into the logistic regression model
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    #train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_pred

#full evaluation function
#evaluates model performance using:
#accuracy
#classification report (precision, recall, f1)
#confusion matric
#shows misclassified examples
def evaluate_predictions(test_df, true_col, pred_col, text_col, dataset_name, version, n_errors=10):

    #true and predicted labels
    y_true = test_df[true_col]
    y_pred = test_df[pred_col]

    #calculate accuracy
    acc = accuracy_score(y_true, y_pred)

    #prints summary
    print("=" * 70)
    print(f"{dataset_name} - {version}")
    print("=" * 70)
    print(f"Accuracy: {acc:.4f}")

    #prints classification matrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # plots confusion matrix
    # make plot size depend on number of labels
    #cm with lots of classes were overlapping

    n_labels = y_true.nunique()

    if n_labels <= 4:
        figsize = (6, 5)
        rotation = 45
    else:
        figsize = (18, 12)
        rotation = 45

    fig, ax = plt.subplots(figsize=figsize)

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=ax,
        xticks_rotation=rotation,
        values_format='d',
        colorbar=False
    )

    ax.set_title(f"{dataset_name} Confusion Matrix ({version})", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)

    # improve label readability
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # give more room at the bottom for long rotated labels
    plt.subplots_adjust(bottom=0.35)

    plt.show()

    #identifies misclassified examples
    errors = test_df[test_df[true_col] != test_df[pred_col]][[text_col, true_col, pred_col]].copy()
    print(f"\nNumber of errors: {len(errors)}")

    #shows first few errors
    if len(errors) > 0:
        print(f"\nFirst {min(n_errors, len(errors))} error examples:")
        print(errors.head(n_errors))
    else:
        print("\nNo errors found.")

    #returns summary as a dictionary
    return {
        "dataset": dataset_name,
        "version": version,
        "accuracy": acc,
        "n_errors": len(errors)
    }
