import pandas as pd
import numpy as np

class Labeller:
    def __init__(self):
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    #feature engineering
    def create_gatekeeper_labels(self, df):
        """
        Creates safe (1) vs not_safe (0) labels.
        Safe: All 6 checkboxes empty.
        Not Safe: Any checkbox ticked.
        """
        # Sum the values across the label columns
        label_sum = df[self.labels].sum(axis=1)
        
        # If sum is 0, it's Safe (1). If sum > 0, it's Not Safe (0).
        df['gatekeeper_label'] = np.where(label_sum == 0, 1, 0)
        return df

    def create_specialist_labels(self, df):
        """
        Creates 3 categories:
        2 (Safe): No checkboxes.
        1 (Offensive): toxic OR obscene (and NOT hate speech).
        0 (Hate Speech): severe_toxic OR threat OR insult OR identity_hate (Override).
        """
        # Define the buckets to be checked for making labels
        hate_speech_cols = ['severe_toxic', 'threat', 'insult', 'identity_hate']
        offensive_cols = ['toxic', 'obscene']
        
        conditions = [
            # Check for Hate Speech first (Override logic)
            (df[hate_speech_cols].sum(axis=1) > 0),
            
            # Check for Offensive next (if not Hate Speech)
            (df[offensive_cols].sum(axis=1) > 0)
        ]
        
        choices = [
            0, # Hate Speech
            1  # Offensive
        ]
        
        # Default is Safe (2)
        df['specialist_label'] = np.select(conditions, choices, default=2)
        return df

if __name__ == "__main__":
    # Test cases
    data = {
        'comment_text': ['safe comment', 'bad word', 'i will kill you'],
        'toxic': [0, 1, 1],
        'severe_toxic': [0, 0, 0],
        'obscene': [0, 1, 0],
        'threat': [0, 0, 1],
        'insult': [0, 0, 0],
        'identity_hate': [0, 0, 0]
    }
    df = pd.DataFrame(data)
    labeller = Labeller()
    
    df = labeller.create_gatekeeper_labels(df)
    df = labeller.create_specialist_labels(df)
    
    print(df[['comment_text', 'gatekeeper_label', 'specialist_label']])
