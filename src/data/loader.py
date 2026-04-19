import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def load_fashion_mnist_from_kaggle():
    """Download Fashion MNIST CSVs from Kaggle using kagglehub."""
    import kagglehub
    path = kagglehub.dataset_download("zalando-research/fashionmnist")
    print(f"Dataset downloaded to: {path}")
    return load_fashion_mnist_from_csv(path)


def load_fashion_mnist_from_csv(data_dir):
    """Load Fashion MNIST from local CSV files."""
    train_df = pd.read_csv(os.path.join(data_dir, 'fashion-mnist_train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'fashion-mnist_test.csv'))
    print(f"Train set: {train_df.shape} | Test set: {test_df.shape}")
    return train_df, test_df


def preprocess_data(train_df, test_df, val_size=0.2, seed=42):
    """Parse CSVs, normalize pixels, and split into train/val/test."""
    y_train_full = train_df['label'].values
    X_train_full = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    y_test = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size, random_state=seed, stratify=y_train_full
    )

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test
