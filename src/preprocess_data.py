import numpy as np
import torch
import os
import openml
import pandas as pd
from sklearn import datasets, model_selection, metrics
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler,MinMaxScaler
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from collections import Counter


def get_columns_with_null_values(df):
    '''
    Identifies and displays the columns in a DataFrame that contains missing data.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - list: A list of column names that contain missing values.
    '''
    columns_with_nan = df.columns[df.isnull().any()].tolist()
    print("Columns with null values:", columns_with_nan)

    df_columns_with_nan = df[columns_with_nan ]
    null_values = df_columns_with_nan.isnull().sum()

    print("Missing values:")
    print(null_values)
    return columns_with_nan


def fill_categorical_nan_with_new_cat(df_var):
    '''
    Adds a new category to a categorical feature when missing data is present.

    Parameters:
    - df_var (pd.Series): The pandas Series representing the categorical feature to modify.

    Returns:
    - pd.Series: The updated Series with an additional category for missing values.
    '''
    df_var = df_var.cat.add_categories(['Unknown'])
    df_var = df_var.fillna('Unknown')
    return df_var


def normalize_numerical_data(df_train, standardscaler_status=True, feature_range_min_max=(0, 1)):
    '''
    Normalizes numerical features in training dataset using either MinMax Scaling or Standard Scaling.
    
    Parameters:
    - df_train (pd.DataFrame): The training dataset.
    - standardscaler (bool): If True, applies Standard Scaling; otherwise, applies MinMax Scaling.
    - feature_range_min_max (int,int): Range of the min value and the max value of MinMax Scaler
    
    Returns:
    -  pd.DataFrame : The normalized training dataset.
    -  scaler: The fitted scaler object based on the training data.
    '''
    if standardscaler_status:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler(feature_range=feature_range_min_max)

    df_train = scaler.fit_transform(df_train)
    return df_train, scaler


def inpute_missing_number_numerical_data(df_train,n_neighbors=10):
    '''
    Imputation for completing missing values using k-Nearest Neighbors (sklearn function)
    
    Parameters:
    - df_train (pd.DataFrame): The training dataset.
    - n_neighbors (int): Number of neighboring samples to use for imputation.
    
    Returns:
    -  pd.DataFrame : The imputed training dataset.
    -  imputer: The fitted imputed object based on the training data.
    '''
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_train = imputer.fit_transform(df_train)
    return df_train, imputer


def check_index_in_range(unicat,df_cat_inputs):
    '''
    Checks whether the categorical indices in `cat_inputs` are within the valid range
    defined by their unique category values.

    Parameters:
    - unicat (list): A list where each element contains the unique values of a 
                     categorical feature.
    - cat_inputs (pd.DataFrame or np.ndarray): A DataFrame or array containing 
                                               categorical feature values.

    Returns:
    - None: Prints an error message if an index exceeds the valid range.
    '''
    for i, num_categories in enumerate(unicat):
        max_val = df_cat_inputs[:, i].max().item()
        if max_val >= num_categories:
            print(f"Error: Column {i} has a maximum value {max_val}, but it's limit is {num_categories - 1}")


def read_dataset_by_id(id):
    '''
    Fetches a dataset from OpenML by its ID and returns a dictionary containing both the 
    dataset and its metadata.
    
    Parameters:
    - id (int). The OpenML dataset ID.
    
    Returns:
    - data (dictionary). A dictionary with the dataset and its associated metadata.
    '''
    dataset_info = openml.datasets.get_dataset(id, download_data=False)
    print('Dataset name: ', dataset_info.name)
    target = dataset_info.default_target_attribute

    features, outputs, categorical_mask, columns = dataset_info.get_data(
            dataset_format="dataframe", target=target)

    # Remove rows with all nans
    features = features.dropna(axis=0, how="all")
    # Remove columns with all nans
    features = features.dropna(axis=1, how="all")

    removed_cols = set(columns) - set(columns).intersection(set(features.columns))

    removed_mask = np.isin(columns, list(removed_cols))
    columns = np.array(columns)
    columns = columns[~removed_mask]

    categorical_mask = np.array(categorical_mask)
    categorical_mask = categorical_mask[~removed_mask]

    assert features.shape[0] == outputs.shape[0], "Invalid features and predictions shapes"

    if outputs.dtypes ==  'object':
        outputs = outputs.astype("category")

    labels = {value: idx for idx, value in enumerate(outputs.unique().categories.values)}
    outputs = outputs.cat.rename_categories(labels).values

    categorical = columns[categorical_mask]
    numerical = columns[~categorical_mask]

    categories = {}

    for col in categorical:
        categories[col] = features[col].dropna().unique().categories.values

    data = {
        "features": features,
        "outputs": outputs,
        "target": target,
        "labels": labels,
        "columns": columns,
        "categorical": categorical,
        "categories": categories,
        "n_categorical": [ len(categories[k] )for k in categories ],
        "numerical": numerical,
        "n_numerical": len(numerical)}
    return data


def X_columns_ordered(df):
    '''
    Given a DataFrame returned by `read_dataset_by_id(ID)`, this function processes the 
    dataset and returns:
    
    Parameters:
    - df (dictionary): A dictionary with the dataset and its associated metadata.
    
    Returns:
    - X_ordered (DataFrame): A DataFrame with features reordered such that all numerical 
                             columns come first, followed by categorical columns.
    - n_numerical (int): The number of numerical features in the dataset.
    '''
    X = df["features"] #features

    categorical_features = df['categorical'].tolist() #name of the categorical features
    numerical_features = df['numerical'].tolist() #name of the numerical features

    # Separate categorical and numerical features
    X_categorical = X[categorical_features]  # Categorical features
    X_numerical = X[numerical_features]      # Numerical features

    X_ordered = pd.concat([X_numerical, X_categorical], axis=1)
    X_categorical = X_categorical.loc[:, X_categorical.nunique() > 1]
    n_numerical = X_numerical.shape[1]
    return X_ordered, n_numerical


def split_data(X, y, seed, test_size):
    '''
    Split dataset into train and test subsets (sklearn function).
    
    Parameters:
    - X (numpy array): Features of the dataset with shape (number_instances, number_features). 
    - y (numpy array): Labels of the dataset (number_instances,).
    - test_size (float): Represents the proportion of the dataset to include in the 
                         test split; should be between 0.0 and 1.0.
    - seed (int): Pseudo-random number generator state used for random sampling.
    
    Returns:
    - X_train (numpy array): Features of the training dataset.
    - X_test (numpy array): Features of the test dataset.
    - y_train (numpy array): Labels of the training dataset.
    - y_test (numpy array): Labels of the test dataset.
    '''
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state= seed, stratify = y)
    return X_train, X_test, y_train, y_test


def preprocess_numerical_train_data(X,n_numerical,standardscaler_status=True,
                                    feature_range_min_max=(0, 1),n_neighbors=10):
    '''
    Preprocess numerical features of the training dataset. Impute missing values with KNN Imputer, then normalize data 
    with StandardScaler or MinMaxScaler.
    
    Parameters:
    - X (numpy array): Features of the training data with shape 
      (number_instances_train, number_features), ordered with numerical features first followed by categorical features.
    - n_numerical (int): Number of numerical features.
    - standardscaler_status (bool): If True, applies StandardScaler; otherwise, applies 
                                    MinMaxScaler.
    - feature_range_min_max (tuple (min_value, max_value)): If normalization is done with 
                    MinMaxScaler, requires the minimum and maximum values (default: (0, 1)).
    - n_neighbors (int): Number of neighboring samples to use for imputation.

    Returns:
    - X_numerical (numpy array): Preprocessed numerical features of the training dataset.
    - scaler: The fitted scaler object based on the training data.
    - imputer: The fitted imputer object based on the training data.
    '''
    X_numerical = X.iloc[:, :n_numerical] # Numerical features from set
    X_numerical, imputer = inpute_missing_number_numerical_data(X_numerical,n_neighbors)
    X_numerical, scaler = normalize_numerical_data(X_numerical,standardscaler_status,feature_range_min_max)
    return X_numerical,imputer,scaler


def preprocess_numerical_test_data(X,n_numerical,scaler,imputer):
    '''
    Preprocess numerical features of the test dataset. Impute missing values using the 
    trained imputer, then normalize data using the trained scaler.
    
    Parameters:
    - X (numpy array): Features of the test dataset with shape 
     (number_instances_test, number_features), ordered with numerical features first followed by categorical features.
    - n_numerical (int): Number of numerical features.
    - scaler: The fitted scaler object based on the training data.
    - imputer: The fitted imputer object based on the training data.

    Returns:
    - X_numerical (numpy array): Preprocessed numerical features of the test dataset.
    '''
    X_numerical = X.iloc[:, :n_numerical] # Numerical features from set
    X_numerical = imputer.transform(X_numerical)
    X_numerical = scaler.transform(X_numerical)
    return X_numerical


def get_unique_categories_of_categorical_features(df,cat_features):
    '''
    Retrieves the unique categories for each categorical feature in a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.
    - cat_features (list): List of column names corresponding to categorical features.
    
    Returns:
    - list: A list with the values of unique categories of each category feature.
    '''
    category_sizes = {col: df[col].nunique() for col in cat_features}
    print('Unique categories: ', category_sizes)
    unicat = list(category_sizes.values())
    return unicat


def preprocess_categorical_train_data(X,n_numerical):
    '''
    Preprocess categorical features for training data. Fill missing values with "unknown" category of each feature, 
    then apply OrdinalEncoder to transform categorical features into numerical format.
    
    Parameters:
    - X (numpy array): Features of the training dataset with shape (number_instances_train, number_features), ordered with 
      numerical features first followed by categorical features.
    - n_numerical (int): Number of numerical features.
    
    Returns:
    - X_categorical (numpy array): Preprocessed categorical features of the training dataset.
    - unicat (list): A list containing the unique categories for each categorical feature.
    - encoder: The fitted encoder object based on the training data.
    '''
    X_categorical = X.iloc[:, n_numerical:] # Categorical features from training set
    cat_features = list(X_categorical.columns)
    get_columns_with_null_values(X_categorical)
    X_categorical = X_categorical.astype(str).fillna("unknown")
    unicat = get_unique_categories_of_categorical_features(X_categorical,cat_features )
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_categorical = encoder.fit_transform(X_categorical)
    return X_categorical, unicat, encoder


def preprocess_categorical_test_data(X,n_numerical,encoder):
    '''
    Preprocess categorical features for test data. Fill missing values with the "unknown" category for each feature, 
    then encode categorical features using the fitted encoder from the training dataset.
    
    Parameters:
    - X (numpy array): Features of the training dataset with shape (number_instances_train, number_features), ordered with 
      numerical features first followed by categorical features.
    - n_numerical (int): Number of numerical features.
    
    Returns:
    -  X_categorical (numpy array): Preprocessed categorical features of the training dataset.
    '''
    X_categorical = X.iloc[:, n_numerical:] # Categorical features from training set
    X_categorical = X_categorical.astype(str).fillna("unknown")
    X_categorical = encoder.transform(X_categorical)
    return X_categorical


def all_preprocess(id, seed, test_size,standardscaler_status=True,feature_range_min_max=(0,1)):
    '''
    Fetches a dataset from OpenML by its ID along with its metadata. Reorders the dataset features to place numerical features first, 
    followed by categorical features. 
    Splits the dataset into training and testing sets, then splits the training set again to create a validation set. 
    If numerical features are present, preprocesses them for training, validation, and test sets. 
    If categorical features are present, preprocesses them for training, validation, and test sets. 
    Encodes the labels using LabelEncoder. 
    Returns preprocessed feature arrays combining numerical and categorical features in that order (concatenating an empty array if 
    one type is missing).
    
    Parameters:
    - id (int): The OpenML dataset ID.
    - seed (int): Pseudo-random number generator state used for random sampling.
    - test_size (float): Represents the proportion of the dataset to include in the test split; should be between 0.0 and 1.0.
    - standardscaler_status (bool): If True, applies StandardScaler; otherwise, applies MinMaxScaler.
    - feature_range_min_max (tuple (min_value, max_value)): If normalization is done with MinMaxScaler, 
      requires the minimum and maximum values (default: (0, 1)).

    Returns:
    - X_train (numpy array): Preprocessed features of the training dataset with shape (number_instances_train, number_features), 
                             ordered with numerical features first followed by categorical features.
    - X_val (numpy array): Preprocessed features of the validation dataset with shape (number_instances_val, number_features), 
                           ordered with numerical features first followed by categorical features.
    - X_test (numpy array): Preprocessed features of the test dataset with shape (number_instances_test, number_features), 
                            ordered with numerical features first followed by categorical features.
    - y_train (numpy array): Encoded labels of the training dataset.
    - y_val (numpy array): Encoded labels of the validation dataset.
    - y_test (numpy array): Encoded labels of the test dataset.
    - numerical_indices (list): List of indices for numerical features.
    - categorical_indices (list): List of indices for categorical features.
    - n_categories (list): List containing the number of unique categories for each categorical feature.
    - n_labels (int): Number of unique labels in the dataset.
    '''
    df = read_dataset_by_id(id)
    X_ordered, n_numerical = X_columns_ordered(df)
    X_train, X_test, y_train, y_test = split_data(X_ordered, df["outputs"].codes, seed=seed, test_size=test_size)
    X_train, X_val, y_train, y_val = split_data(X_train, y_train, seed=seed, test_size=test_size)

    # Numerical data
    if n_numerical > 0:
        X_train_num,imputer,scaler = preprocess_numerical_train_data(X_train,n_numerical,standardscaler_status,feature_range_min_max)
        X_val_num = preprocess_numerical_test_data(X_val,n_numerical,scaler,imputer)
        X_test_num = preprocess_numerical_test_data(X_test,n_numerical,scaler,imputer)
    else:
        X_train_num = np.empty((X_train.shape[0], 0))
        X_val_num = np.empty((X_val.shape[0], 0))
        X_test_num = np.empty((X_test.shape[0], 0))

    # Categorical data
    if X_train.shape[1] > n_numerical:
        X_train_cat,n_categories,encoder = preprocess_categorical_train_data(X_train,n_numerical)
        X_val_cat = preprocess_categorical_test_data(X_val,n_numerical,encoder)
        X_test_cat = preprocess_categorical_test_data(X_test,n_numerical,encoder)

        for col_idx in range(X_val_cat.shape[1]):
            if (-1.0 in X_val_cat[:, col_idx]) | (-1.0 in X_test_cat[:, col_idx]):
                # Recorrer orden para iniciar con 0
                X_train_cat[:, col_idx] += 1
                X_val_cat[:, col_idx] += 1
                X_test_cat[:, col_idx] += 1
                # Agregar una categoria a categorias unicas
                n_categories[col_idx] = n_categories[col_idx] + 1
    else:
        X_train_cat = np.empty((X_train.shape[0], 0))
        X_val_cat = np.empty((X_val.shape[0], 0))
        X_test_cat = np.empty((X_test.shape[0], 0))
        n_categories = []

    # Target
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    # Concatenate numerical and categorical features for datasets
    X_train = np.concatenate([X_train_num, X_train_cat], axis=1)
    X_val = np.concatenate([X_val_num, X_val_cat], axis=1)
    X_test = np.concatenate([X_test_num, X_test_cat], axis=1)

    # List of indices for numerical features
    numerical_indices = list(range(n_numerical))
     # List of indices for categorical features
    categorical_indices = list(range(n_numerical, X_train.shape[1]))

    n_labels = len(np.unique(y_train))

    return X_train,X_val,X_test,y_train,y_val,y_test,numerical_indices,categorical_indices,n_categories,n_labels

