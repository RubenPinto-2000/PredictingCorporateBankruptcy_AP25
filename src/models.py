"""
This module defines and trains the baseline and non-linear models chosen and used in this project
to compare different classification approaches on the Polish Bankruptcy dataset.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# random_state=42 ensures reproducibility and is a conventional value in machine learning
# We use an 80/20 train/test split to train the models

def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train a logistic regression model to learn linear patterns between financial ratios and bankruptcy.
    This is the chosen linear baseline model.
    Used as a linear baseline with class imbalance handling.
    """
    # Setting max_iter to 1000 allows the model to converge
    # class_weight="balanced" compensates for the huge imbalance between the number of bankruptcies and non-bankruptcies
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    model.fit(X_train, y_train)  # learns the relationships between features and target
    return model  # can now make predictions


def train_random_forest(X_train, y_train, random_state=42):
    """
    Train a random forest classifier to learn non-linear patterns between financial ratios and bankruptcy.
    Used to capture non-linear relationships between features.
    """
    # n_estimators=300 permits enough trees to stabilize predictions without excessive computation
    # class_weight="balanced_subsample" applies class balancing at each bootstrap sample level, 
    # which works better with bagging than global "balanced" weighting
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train a k-nearest neighbors classifier on training data so it stores enough examples to make 
    comparisons and predictions based on similarity.
    Distance-based baseline model.
    """
    # n_neighbors=5 is small enough to capture local patterns, large enough to smooth out noise
    # This represents a good compromise for the number of neighbors
    # This is a common default that works well
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)  # stores the training data for later similarity-based predictions
    return model

