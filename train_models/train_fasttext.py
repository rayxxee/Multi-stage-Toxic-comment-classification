import pandas as pd
import yaml
import os
import joblib
from sklearn.model_selection import train_test_split
from src.utils import set_seed
from src.data.cleaner import TextCleaner
from src.data.labeller import Labeller
from src.models.fasttext_model import FastTextClassifier

def main():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    set_seed(config['random_seed'])
    
    df = pd.read_csv(config['paths']['raw_data'])
    df = TextCleaner(regex_pattern=config['preprocessing']['regex_pattern']).process_dataframe(df)
    labeller = Labeller()
    df = labeller.create_gatekeeper_labels(df)
    
    df_train, _ = train_test_split(df, test_size=config['training']['test_size'], random_state=config['random_seed'], stratify=df['gatekeeper_label'])
    
    X_train = df_train['comment_text'].tolist()
    y_train = df_train['gatekeeper_label'].tolist()
    
    ft_params = config['model_params']['fasttext']
    model = FastTextClassifier(
        lr=ft_params['lr'], epoch=ft_params['epoch'], wordNgrams=ft_params['wordNgrams'],
        minn=ft_params['minn'], maxn=ft_params['maxn'], dim=ft_params['dim'], loss=ft_params['loss']
    )
    print("Training FastText...")
    model.train(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fasttext.joblib')
    print("Model saved to models/fasttext.joblib")

if __name__ == "__main__":
    main()
