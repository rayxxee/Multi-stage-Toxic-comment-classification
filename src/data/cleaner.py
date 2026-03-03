import pandas as pd
import re
import numpy as np

class TextCleaner:
    def __init__(self, regex_pattern="[^a-z0-9 ]"):
        self.regex_pattern = regex_pattern

    def clean_text(self, text):
        """
        Applies caching strategies:
        1. Typecast to string
        2. Lowercase
        3. Remove noise (regex)
        4. Strip whitespace
        """
        # 1. Typecast to string
        text = str(text)
        
        # 2. Lowercasing
        text = text.lower()
        
        # 3. Stripping Out "Noise"
        # Keep only lowercase letters (a-z), numbers (0-9), and spaces
        text = re.sub(self.regex_pattern, '', text)
        
        # 4. Final Trimming
        # Remove extra spaces and strip leading/trailing whitespace
        text = re.sub(' +', ' ', text).strip()
        
        return text

    def process_dataframe(self, df, text_column='comment_text'):
        print(f"Cleaning {len(df)} rows...")
        
        # Apply cleaning
        df[text_column] = df[text_column].apply(self.clean_text)
        
        # 4. Taking Out the Trash (Empty & Useless Comments)
        # Drop rows where comment is empty string
        initial_count = len(df)
        df = df[df[text_column] != '']
        final_count = len(df)
        
        print(f"Removed {initial_count - final_count} empty rows.")
        return df

if __name__ == "__main__":
    # verification example
    cleaner = TextCleaner()
    raw = "YOU are a terrible person!!! >:("
    print(f"Raw: '{raw}'")
    print(f"Clean: '{cleaner.clean_text(raw)}'")
