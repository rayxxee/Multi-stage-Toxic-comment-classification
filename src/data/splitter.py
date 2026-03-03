import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df, target_column):
        """
        Splits data into train and validation sets with stratification.
        """
        print(f"Splitting data with test_size={self.test_size}...")
        
        X = df['comment_text']
        y = df[target_column]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        # Check for leakage (intersection of indices)
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        intersection = train_indices.intersection(val_indices)
        
        if intersection:
            raise ValueError(f"Data Leakage Detected! {len(intersection)} indices overlap.")
        else:
            print("Leakage Check Passed: No intersection between train and validation indices.")
            
        return X_train, X_val, y_train, y_val
