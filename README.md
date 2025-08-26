# Digit-recognition-knn-vs-tree

🧠 Handwritten Digit Classifier: KNN vs Decision Tree
This project compares the performance of two classic machine learning algorithms — K-Nearest Neighbors (KNN) and Decision Tree Classifier — on the MNIST Digit Recognition Dataset.

The program:

Trains both classifiers on a portion of the dataset

Tests their performance on unseen data

Reports accuracy and identifies misclassified samples

Allows the user to visualize predictions of specific test digits

🔧 Features
Reads and processes the MNIST data (train.csv, test.csv)

Trains a DecisionTreeClassifier and KNeighborsClassifier from scikit-learn

Outputs individual and shared misclassifications

Interactive CLI tool to view specific digit predictions

Displays digit images with predicted vs actual values

📁 Data Source
Download the dataset from Kaggle:
https://www.kaggle.com/competitions/digit-recognizer

🛠 Requirements
Python 3

Libraries: numpy, pandas, matplotlib, scikit-learn
