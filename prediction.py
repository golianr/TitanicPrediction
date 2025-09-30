import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def predict():
    dataset = pd.read_csv("Titanic-Dataset.csv")
    
    # Cleaning dataset
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})
    dataset = dataset[["Survived", "Age", "Sex"]]
    dataset["Sex"] = dataset["Sex"].fillna(dataset["Sex"].mean())
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())


    if dataset.isnull().sum().sum() != 0:
        print("Missing values")
        return

    # Splitting dataset
    test_size = 0.2
    split_idx = int(len(dataset) * (1 - test_size))
    train = dataset[:split_idx]
    test = dataset[split_idx:]

    # Using Logistic regression
    model = LogisticRegression()

    features_train = train[["Age", "Sex"]]
    target_train = train["Survived"]

    features_test = test[["Age", "Sex"]]
    target_test = test[["Survived"]]

    model.fit(features_train, target_train)
    prediction = model.predict(features_test)

    # Analyzing results
    print("Accuracy: ", accuracy_score(target_test, prediction))
    print("Confusion Matrix: ", confusion_matrix(target_test, prediction))
    print("Classification Report:\n", classification_report(target_test, prediction))

    # Creating graphs
    plt.figure(figsize=(10,5))
    plt.plot(range(len(target_test)), target_test.values, label='True', color='blue', alpha=0.6)
    plt.plot(range(len(prediction)), prediction, label='Predicted', color='red', linestyle='--', alpha=0.4)
    plt.xlabel('Sample index')
    plt.ylabel('Survived')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,5))
    plt.scatter(range(len(target_test)), target_test, label='True', color='blue', alpha=0.6)
    plt.scatter(range(len(prediction)), prediction, label='Predicted', color='red', alpha=0.4)
    plt.xlabel('Sample index')
    plt.ylabel('Survived')
    plt.title('True vs Predicted Values (scatter)')
    plt.legend()
    plt.show()

    return prediction

predict()