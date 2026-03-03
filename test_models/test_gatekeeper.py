import pandas as pd
import yaml
import joblib
import os
from sklearn.model_selection import train_test_split
from src.utils import set_seed
from src.data.cleaner import TextCleaner
from src.data.labeller import Labeller
from src.evaluation.metrics import compute_metrics
from src.evaluation.reporter import ModelReporter

def main():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    set_seed(config['random_seed'])
    
    df = pd.read_csv(config['paths']['raw_data'])
    df = TextCleaner(regex_pattern=config['preprocessing']['regex_pattern']).process_dataframe(df)
    labeller = Labeller()
    df = labeller.create_gatekeeper_labels(df)
    df = labeller.create_specialist_labels(df)
    
    _, df_val = train_test_split(df, test_size=config['training']['test_size'], random_state=config['random_seed'], stratify=df['specialist_label'])
    
    X_val = df_val['comment_text'].tolist()
    y_val = df_val['gatekeeper_label'].tolist()
    
    print("Loading Gatekeeper from models/gatekeeper.joblib...")
    model = joblib.load('models/gatekeeper.joblib')
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    
    binary_labels = {int(k): v for k, v in config['labels']['binary'].items()}
    metrics = compute_metrics(y_val, y_pred, y_proba, binary_labels)
    
    reporter = ModelReporter()
    reporter.report_model("Gatekeeper", None, metrics)

if __name__ == "__main__":
    main()
