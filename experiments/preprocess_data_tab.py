import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder 


def create_dict_of_the_data(X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                            task_type,
                            n_cont_features,
                            n_categorical_features,
                            num_idx=None,
                            cat_idx=None):
    '''
    Create a dictionary containing the data splits (train, val, and test).  
    Each split includes numerical features, categorical features, and labels.  
    All data are converted to PyTorch tensors.

    Arguments:
    - X_train (numpy array): Training data with shape (num_instances, num_features)
    - y_train (numpy array): Training labels with shape (num_instances,)
    - X_val (numpy array): Validation data with shape (num_instances, num_features)
    - y_val (numpy array): Validation labels with shape (num_instances,)
    - X_test (numpy array): Test data with shape (num_instances, num_features)
    - y_test (numpy array): Test labels with shape (num_instances,)
    - task_type (str): Type of classification task. Valid options are ["binary", "multiclass"]
    - n_cont_features (int): Number of numerical features
    - n_categorical_features (int): Number of categorical features
    - num_idx (list[int]): List of indices corresponding to numerical features
    - cat_idx (list[int]): List of indices corresponding to categorical features

    Returns:
    - dataset (dict): Dictionary containing PyTorch tensors for train, val, and test splits.
    Each split includes numerical features, categorical features, and labels.
    '''
    data_numpy = {"train": {"y": y_train},
                  "val": {"y": y_val},
                  "test": {"y": y_test}}

    if n_cont_features != 0:
        data_numpy["train"]["x_cont"] = X_train[:, num_idx].astype(np.float32)
        data_numpy["val"]["x_cont"] = X_val[:, num_idx].astype(np.float32)
        data_numpy["test"]["x_cont"] = X_test[:, num_idx].astype(np.float32)

    if n_categorical_features != 0:
        data_numpy["train"]["x_cat"] = (X_train[:, cat_idx]).astype(np.int64)
        data_numpy["val"]["x_cat"] = (X_val[:, cat_idx]).astype(np.int64)
        data_numpy["test"]["x_cat"] = (X_test[:, cat_idx]).astype(np.int64)

    # >>> Convert data to tensors.
    dataset = {
        part: {k: torch.as_tensor(v) for k, v in data_numpy[part].items()}
        for part in data_numpy
    }

    if task_type != "multiclass":
        # Required by F.binary_cross_entropy_with_logits
        for part in dataset:
            dataset[part]["y"] = dataset[part]["y"].float()

    return dataset


def apply_scaler(num_train_instances):
    '''
    Create and return a QuantileTransformer scaler.
    The number of quantiles is set to the minimum between 1000 and the number
    of training instances. (Tabzilla)

    Arguments:
        - num_train_instances (int): Number of training instances
    
    Returns:
        - scaler_function (QuantileTransformer): Initialized quantile scaler
    '''
    # Use either 1000 quantiles or num. training instances, whichever is smaller
    scaler_function = QuantileTransformer(n_quantiles=min(num_train_instances, 1000))  
    return scaler_function 


def preprocess_categorical_test_data(X_categorical,
                                     encoder):
    '''
    Preprocess categorical data.
    Missing values are handled by assigning them to an additional category
    (e.g., "unknown"). The data are then encoded using the encoder fitted on the training set. 
    Ordinal-encoded values are shifted by 1 so that missing values are assigned to 0 and 
    the remaining categories are mapped to positive integers.

    Arguments:
    - X_categorical (numpy array): Numpy array containing the categorical features of the 
      validation or test dataset
    - encoder: Encoder fitted on the training data

    Returns:
    - X_encoded (numpy array): Validation or test array with preprocessed categorical features
    '''
    X_categorical = pd.DataFrame(X_categorical)
    X_categorical = X_categorical.astype(str).fillna("unknown")
    X_encoded = encoder.transform(X_categorical)
    # Shifting: -1 -> 0, 0 -> 1, etc.
    X_encoded = X_encoded + 1
    return X_encoded


def preprocess_categorical_train_data(X_categorical):
    '''
    Preprocess categorical data for the training set.
    Missing values are handled by assigning them to an additional category (e.g., "unknown"). 
    The data are then fitted and transformed using an ordinal encoder. 
    Ordinal-encoded values are shifted by 1 so that missing values are assigned to 0 and the 
    remaining categories are mapped to positive integers.

    Arguments:
    - X_categorical (numpy array): Numpy array containing the categorical features of the 
    training dataset

    Returns:
    - X_encoded (numpy array): Training array with preprocessed categorical features
    - n_categories (list[int]): List containing the number of categories for
      each feature (original categories + 1 to handle the unknown category)
    - encoder: Encoder fitted on the training data
    '''
    X_categorical = pd.DataFrame(X_categorical)
    X_categorical = X_categorical.astype(str).fillna("unknown")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_encoded = encoder.fit_transform(X_categorical)

    # Shifting 
    X_encoded = X_encoded + 1
    
    # Calculate number of categories: original + 1 (to handle unknown category)
    n_categories = [int(len(cats) + 1) for cats in encoder.categories_]
    return X_encoded, n_categories, encoder


def preprocess_all_data(dataset, 
                        train_index, 
                        val_index,
                        test_index, 
                        impute=True, 
                        scaler=True):
    '''
    Preprocess numerical and categorical features for the training, validation,
    and test datasets, as well as their corresponding labels.

    Arguments:
    - dataset (tabzilla_datasets.TabularDataset): Input tabular dataset
    - train_index (list[int]): List of indices for the training data
    - val_index (list[int]): List of indices for the validation data
    - test_index (list[int]): List of indices for the test data
    - impute (bool): If True, impute missing values in numerical features
    - scaler (bool): If True, scale numerical features using a QuantileTransformer

    Returns:
    - X_train (numpy array): Preprocessed training data
    - X_val (numpy array): Preprocessed validation data
    - X_test (numpy array): Preprocessed test data
    - y_train (numpy array): Training labels
    - y_val (numpy array): Validation labels
    - y_test (numpy array): Test labels
    - numerical_indices (list[int]): List of indices corresponding to numerical features
    - categorical_indices (list[int]): List of indices corresponding to categorical features
    - n_categories (list[int]): List containing the number of categories per categorical feature
    - n_labels (int): Number of labels
    '''
    # Scale
    if scaler:      
        scaler_function  = apply_scaler(len(train_index))
    
    # Create  list with 1 when numerical features and 0 for categorical features
    num_mask = np.ones(dataset.X.shape[1], dtype=int)
    num_mask[dataset.cat_idx] = 0

    X_train, y_train = dataset.X[train_index], dataset.y[train_index]
    X_val, y_val = dataset.X[val_index], dataset.y[val_index]
    X_test, y_test = dataset.X[test_index], dataset.y[test_index]

    # Impute numerical features
    num_idx = np.where(num_mask)[0] # Get numeric features index
    cat_idx = np.where(num_mask==0)[0]
    
    if num_idx.shape[0]!=0:
        if impute:
            # The imputer drops columns that are fully NaN. So, we first identify columns that are fully NaN and set them to
            # zero. This will effectively drop the columns without changing the column indexing and ordering that many of
            # the functions in this repository rely upon.
            fully_nan_num_idcs = np.nonzero(
            (~np.isnan(X_train[:, num_idx].astype("float"))).sum(axis=0) == 0)[0]
            if fully_nan_num_idcs.size > 0:
                X_train[:, num_idx[fully_nan_num_idcs]] = 0
                X_val[:, num_idx[fully_nan_num_idcs]] = 0
                X_test[:, num_idx[fully_nan_num_idcs]] = 0
        
            # Impute numerical features, and pass through the rest
            numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer())])
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, num_idx),
                    ("pass", "passthrough", cat_idx),],
            )
            X_train = preprocessor.fit_transform(X_train)
            X_val = preprocessor.transform(X_val)
            X_test = preprocessor.transform(X_test)

            # Re-order columns (ColumnTransformer permutes them)
            # First columns  (numerical features) then (categorical features)
            perm_idx = []
            running_num_idx = 0
            running_cat_idx = 0
            for is_num in num_mask:
                if is_num > 0:
                    perm_idx.append(running_num_idx)
                    running_num_idx += 1
                else:
                    perm_idx.append(running_cat_idx + len(num_idx))
                    running_cat_idx += 1
            X_train = X_train[:, perm_idx]
            X_val = X_val[:, perm_idx]
            X_test = X_test[:, perm_idx]

            num_idx = np.where(num_mask == 1)[0]
            cat_idx = np.where(num_mask == 0)[0]

            # Scale with Quantile
            if scaler:
                print(f"Scaling the data using {scaler}...")
                X_train[:, num_idx] = scaler_function.fit_transform(X_train[:, num_idx])
                X_val[:, num_idx] = scaler_function.transform(X_val[:, num_idx])
                X_test[:, num_idx] = scaler_function.transform(X_test[:, num_idx])


    if len(cat_idx) !=0:
        X_categorical_train = X_train[:, cat_idx]
        X_categorical_test = X_test[:, cat_idx]
        X_categorical_val = X_val[:, cat_idx]
        X_train_cat, n_categories, encoder = preprocess_categorical_train_data(X_categorical_train)
        X_val_cat = preprocess_categorical_test_data(X_categorical_val,encoder)
        X_test_cat = preprocess_categorical_test_data(X_categorical_test,encoder)
        
        X_train[:, cat_idx] = X_train_cat.astype(int)
        X_val[:, cat_idx] = X_val_cat.astype(int)
        X_test[:, cat_idx] = X_test_cat.astype(int)

        # Forzar conversión a long/int para evitar errores de indexación en PyTorch
        #X_train = X_train.astype(float) # Las numéricas suelen ser float
    else:
        n_categories = []

    # Target
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    
    # List of indices for numerical and categoricalfeatures
    numerical_indices = list(num_idx)
    categorical_indices = list(cat_idx)
    n_labels = len(np.unique(y_train))
    numerical_indices= [int(x) for x in numerical_indices]
    categorical_indices = [int(x) for x in categorical_indices]
    return X_train,X_val,X_test,y_train,y_val,y_test,numerical_indices,categorical_indices,n_categories,n_labels
