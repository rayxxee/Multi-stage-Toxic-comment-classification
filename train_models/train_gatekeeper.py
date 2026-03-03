import pandas as pd
import yaml
import os
import joblib
from sklearn.model_selection import train_test_split
from src.utils import set_seed
from src.data.cleaner import TextCleaner
from src.data.labeller import Labeller
from src.models.gatekeeper_lr import GatekeeperLR

def main():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    set_seed(config['random_seed'])
    
    df = pd.read_csv(config['paths']['raw_data'])
    df = TextCleaner(regex_pattern=config['preprocessing']['regex_pattern']).process_dataframe(df)
    labeller = Labeller()
    df = labeller.create_gatekeeper_labels(df)
    df = labeller.create_specialist_labels(df)
    
    df_train, _ = train_test_split(df, test_size=config['training']['test_size'], random_state=config['random_seed'], stratify=df['specialist_label'])
    
    X_train = df_train['comment_text'].tolist()
    y_train = df_train['gatekeeper_label'].tolist()
    
    gk_params = config['model_params']['gatekeeper_lr']
    model = GatekeeperLR(
        max_features=gk_params['tfidf_max_features'],
        ngram_range=tuple(gk_params['tfidf_ngram_range']),
        max_iter=gk_params['lr_max_iter'],
        C=gk_params['lr_C'],
        random_state=config['random_seed']
    )
    print("Training Gatekeeper...")
    model.train(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/gatekeeper.joblib')
    print("Model saved to models/gatekeeper.joblib")

if __name__ == "__main__":
    main()
