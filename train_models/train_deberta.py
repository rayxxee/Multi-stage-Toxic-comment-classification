import pandas as pd
import yaml
import os
import joblib
from sklearn.model_selection import train_test_split
from src.utils import set_seed
from src.data.cleaner import TextCleaner
from src.data.labeller import Labeller
from src.models.deberta_model import DeBERTaClassifier

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
    y_train = df_train['specialist_label'].tolist()
    
    max_deb = config['training'].get('max_samples_deberta', len(X_train))
    deberta_train_idx = min(len(X_train), max_deb)
    X_train = X_train[:deberta_train_idx]
    y_train = y_train[:deberta_train_idx]

    db_params = config['model_params']['deberta']
    model = DeBERTaClassifier(
        model_name=db_params['model_name'], max_length=db_params['max_length'],
        batch_size=db_params['batch_size'], num_epochs=db_params['num_epochs'],
        learning_rate=db_params['learning_rate']
    )
    print("Training DeBERTa...")
    model.train(X_train, y_train, output_dir=os.path.join(config['paths']['model_output'], "deberta_chkpts"))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/deberta.joblib')
    print("Model saved to models/deberta.joblib")

if __name__ == "__main__":
    main()
