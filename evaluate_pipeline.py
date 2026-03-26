import pandas as pd
import yaml
import os
import joblib
from src.utils import set_seed, setup_logger
from src.data.cleaner import TextCleaner
from src.data.labeller import Labeller
from src.data.splitter import DataSplitter

from src.models.gatekeeper_lr import GatekeeperLR
from src.models.fasttext_model import FastTextClassifier
from src.pipeline.pipeline import HierarchicalPipeline
from src.evaluation.metrics import compute_metrics
from src.evaluation.reporter import ModelReporter

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    set_seed(config['random_seed'])
    
    logger = setup_logger('main_logger', 'pipeline.log')
    logger.info("Hierarchical Pipeline started.")
    
    # 1. Load Data
    raw_path = config['paths']['raw_data']
    try:
        df = pd.read_csv(raw_path)
        # Limit data for DeBERTa training speed if specified
        max_samples = config['training'].get('max_samples', None)
        if max_samples:
            df = df.sample(n=min(len(df), max_samples), random_state=config['random_seed'])
        logger.info(f"Loaded data from {raw_path}. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"File not found: {raw_path}")
        return

    # Cleaner
    cleaner = TextCleaner(regex_pattern=config['preprocessing']['regex_pattern'])
    df = cleaner.process_dataframe(df)
    
    # Labeller
    labeller = Labeller()
    df = labeller.create_gatekeeper_labels(df)
    df = labeller.create_specialist_labels(df)
    logger.info("Data cleaned and labeled.")

    binary_df = df[['comment_text', 'gatekeeper_label']]
    multi_df = df[['comment_text', 'specialist_label']]
    os.makedirs(os.path.dirname(config['paths']['processed_binary']), exist_ok=True)
    binary_df.to_csv(config['paths']['processed_binary'], index=False)
    multi_df.to_csv(config['paths']['processed_multi'], index=False)

    # 2. Split Data
    splitter = DataSplitter(test_size=config['training']['test_size'], random_state=config['random_seed'])
    # X corresponds to comment_text, y is not directly extracted by splitter, it returns a tuple of series 
    # Actually, splitter returns X_train, X_val, y_train, y_val according to standard sklearn train_test_split.
    # In the existing main.py: X_train, X_val, y_train, y_val = splitter.split_data(df, target_col)
    # We need both labels. We will split the whole dataframe.
    df_train, df_val = splitter.split_data(df, target_col='specialist_label', return_df=True) if hasattr(splitter, 'return_df') else (None, None)
    
    # If the standard DataSplitter doesn't support returning df, we do it manually safely
    from sklearn.model_selection import train_test_split
    df_train, df_val = train_test_split(df, test_size=config['training']['test_size'], random_state=config['random_seed'], stratify=df['specialist_label'])
    
    X_train = df_train['comment_text'].tolist()
    y_train_bin = df_train['gatekeeper_label'].tolist()
    y_train_multi = df_train['specialist_label'].tolist()
    
    X_val = df_val['comment_text'].tolist()
    y_val_bin = df_val['gatekeeper_label'].tolist()
    y_val_multi = df_val['specialist_label'].tolist()

    logger.info(f"Data split. Train shape: {len(X_train)}, Val shape: {len(X_val)}")

    # 3. Load Models
    models_dir = config['paths']['model_output']
    
    # --- Gatekeeper LR ---
    gatekeeper_path = os.path.join(models_dir, 'gatekeeper.joblib')
    if not os.path.exists(gatekeeper_path):
        logger.error(f"Model not found: {gatekeeper_path}. Please run train.py first.")
        return
    logger.info("Loading GatekeeperLR...")
    gatekeeper = GatekeeperLR.load(gatekeeper_path)

    # --- FastText ---
    fasttext_path = os.path.join(models_dir, 'fasttext.joblib')
    if not os.path.exists(fasttext_path):
        logger.error(f"Model not found: {fasttext_path}. Please run train.py first.")
        return
    logger.info("Loading FastText...")
    fasttext_model = joblib.load(fasttext_path)

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    import numpy as np

    def plot_model_metrics(y_true, y_prob, y_pred, model_name, label_dict):
        os.makedirs('plots', exist_ok=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[label_dict[i] for i in sorted(label_dict.keys())],
                    yticklabels=[label_dict[i] for i in sorted(label_dict.keys())])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_cm.png')
        plt.close()

        # ROC Curve & PR Curve (if probabilities provided)
        if y_prob is not None and len(y_prob) > 0 and y_prob.ndim == 2 and y_prob.shape[1] == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                plt.figure(figsize=(6,5))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.title(f'{model_name} - ROC Curve')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(f'plots/{model_name}_roc.png')
                plt.close()

                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob[:, 1])
                plt.figure(figsize=(6,5))
                plt.plot(recall_vals, precision_vals, color='blue', lw=2, label='PR Curve')
                plt.title(f'{model_name} - Precision-Recall Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.legend(loc="lower left")
                plt.tight_layout()
                plt.savefig(f'plots/{model_name}_pr.png')
                plt.close()
            except Exception as e:
                logger.error(f"Could not print ROC/PR curve for {model_name}: {e}")

    # Initialize reporter and labels
    reporter = ModelReporter()
    binary_labels = {int(k): v for k, v in config['labels']['binary'].items()}

    # --- Evaluating Gatekeeper Individually ---
    logger.info("Evaluating Gatekeeper individually...")
    gk_labels_pred = gatekeeper.predict(X_val)
    gk_probs = gatekeeper.predict_proba(X_val)
    gk_metrics = compute_metrics(y_val_bin, gk_labels_pred, gk_probs, binary_labels)
    reporter.report_model("Gatekeeper (Individual)", gk_metrics, gk_metrics)
    plot_model_metrics(y_val_bin, gk_probs, gk_labels_pred, "Gatekeeper_Individual", binary_labels)

    # --- Evaluating FastText Individually ---
    logger.info("Evaluating FastText individually...")
    ft_labels_pred = fasttext_model.predict(X_val)
    ft_probs = fasttext_model.predict_proba(X_val)
    ft_metrics = compute_metrics(y_val_bin, ft_labels_pred, ft_probs, binary_labels)
    reporter.report_model("FastText (Individual)", ft_metrics, ft_metrics)
    plot_model_metrics(y_val_bin, ft_probs, ft_labels_pred, "FastText_Individual", binary_labels)

    # --- Initial Error Analysis ---
    logger.info("Performing Error Analysis & logging subset to CSV...")
    errors_gk_mask = (np.array(y_val_bin) != np.array(gk_labels_pred))
    df_errors_gk = pd.DataFrame({
        'text': np.array(X_val)[errors_gk_mask],
        'true_label': np.array(y_val_bin)[errors_gk_mask],
        'pred_label': np.array(gk_labels_pred)[errors_gk_mask]
    })
    df_errors_gk.to_csv('plots/gatekeeper_errors.csv', index=False)

    # 4. Build Pipeline
    # Using the probability threshold from config.yaml for Gatekeeper (safe)
    # The rest gets forwarded to FastText (which doesn't need a threshold, everything passes)
    available_stages = {
        'gatekeeper_lr': {'name': 'gatekeeper_lr', 'model': gatekeeper, 'threshold': config['thresholds']['SAFE_THRESHOLD'], 'type': 'binary', 'pass_label': 1},
        'fasttext': {'name': 'fasttext', 'model': fasttext_model, 'threshold': config['thresholds']['FASTTEXT_THRESHOLD'], 'type': 'binary', 'pass_label': None}
    }
    
    pipeline_stages = [available_stages['gatekeeper_lr'], available_stages['fasttext']]
    pipeline = HierarchicalPipeline(stages=pipeline_stages)

    # 5. Run & Evaluate
    logger.info("Running binary pipeline on validation set...")
    y_final_pred, per_stage_data = pipeline.run(X_val, y_true=y_val_bin)
    
    for stage_name, data in per_stage_data.items():
        m_subset = compute_metrics(data['y_true_subset'], data['y_pred_subset'], data['y_proba_subset'], binary_labels)
        m_full = compute_metrics(data['y_true_full'], data['y_pred_full'], data['y_proba_full'], binary_labels)
        reporter.report_model(f"Pipeline Stage - {stage_name}", m_subset, m_full)

    sys_metrics = compute_metrics(y_val_bin, y_final_pred, None, binary_labels)
    reporter.report_pipeline_summary(per_stage_data, sys_metrics)
    plot_model_metrics(y_val_bin, None, y_final_pred, "Pipeline_Overall", binary_labels)

    # Overall Pipeline Error Analysis
    errors_pipe_mask = (np.array(y_val_bin) != np.array(y_final_pred))
    df_errors_pipe = pd.DataFrame({
        'text': np.array(X_val)[errors_pipe_mask],
        'true_label': np.array(y_val_bin)[errors_pipe_mask],
        'pred_label': np.array(y_final_pred)[errors_pipe_mask]
    })
    df_errors_pipe.to_csv('plots/pipeline_errors.csv', index=False)

    logger.info("Pipeline evaluation and Error Analysis complete. Graphics saved in 'plots/'.")

if __name__ == "__main__":
    main()
