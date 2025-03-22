from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Example: Using Random Forest as the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_slices_performance(df, feature, model, encoder, lb, label_column="salary", categorical_features=None):
    """
    Computes performance metrics (precision, recall, F-beta) for slices of the data
    based on a fixed value of a given categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        The full DataFrame containing both features and label.
    feature : str
        The name of the categorical feature on which to slice.
    model : trained model
        The trained ML model used for inference.
    encoder : sklearn.preprocessing.TransformerMixin
        Encoder used to transform categorical features.
    lb : sklearn.preprocessing.LabelBinarizer or similar
        Binarizer used to transform the label.
    label_column : str, optional
        The name of the label column. Default is "salary".
    categorical_features : list, optional
        The list of categorical features used during training. If not provided,
        the encoder's stored feature names will be used.

    Returns
    -------
    slice_results : list of str
        A list of strings, each describing the performance metrics for one slice.
    """
    slice_results = []
    unique_values = df[feature].unique()
    
    # Use provided categorical_features or default to encoder's feature names
    if categorical_features is None:
        categorical_features = list(encoder.feature_names_in_)
    
    for value in unique_values:
        # Slice the data where the feature equals the current value
        df_slice = df[df[feature] == value]

        # Separate features and label
        y_slice = df_slice[label_column]
        df_features = df_slice.drop(columns=[label_column])
        
        # Separate categorical and continuous features
        X_cat = df_features[categorical_features]
        X_cont = df_features.drop(columns=categorical_features)
        
        # Transform categorical features using the fitted encoder
        X_cat_encoded = encoder.transform(X_cat)
        
        # Concatenate continuous features (if any) with the encoded categorical features.
        # This must match the processing done in process_data.
        if X_cont.shape[1] > 0:
            X_final = np.concatenate([X_cont.to_numpy(), X_cat_encoded], axis=1)
        else:
            X_final = X_cat_encoded

        # Transform the label
        y_slice_encoded = lb.transform(y_slice.values).ravel()

        # Run inference and compute metrics
        preds = inference(model, X_final)
        precision, recall, fbeta = compute_model_metrics(y_slice_encoded, preds)
        result_str = (f"Feature: {feature}, Value: {value} | "
                      f"Precision: {precision:.4f} | "
                      f"Recall: {recall:.4f} | "
                      f"F-beta: {fbeta:.4f}")
        print(result_str)
        slice_results.append(result_str)

    with open("slice_output.txt", "w") as f:
        for line in slice_results:
            f.write(line + "\n")         
