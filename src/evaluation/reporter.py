import pandas as pd

class ModelReporter:
    @staticmethod
    def report_model(model_name, metrics_on_subset, metrics_overall):
        print("="*60)
        print(f"REPORT FOR STAGE: {model_name}")
        print("="*60)
        
        def print_metrics(title, m):
            print(f"\n--- {title} ---")
            if not m:
                print("No samples evaluated.")
                return
            
            print(f"Accuracy:        {m.get('accuracy', 0):.4f}")
            print(f"F1 (Weighted):   {m.get('f1_weighted', 0):.4f}")
            print(f"F1 (Macro):      {m.get('f1_macro', 0):.4f}")
            print(f"Precision:       {m.get('precision', 0):.4f}")
            print(f"Recall:          {m.get('recall', 0):.4f}")
            if m.get('roc_auc') is not None:
                print(f"ROC-AUC:         {m.get('roc_auc', 0):.4f}")
                
            if 'confusion_matrix' in m and 'label_order' in m:
                print("\nConfusion Matrix:")
                df_cm = pd.DataFrame(m['confusion_matrix'], index=m['label_order'], columns=m['label_order'])
                print(df_cm)
            
        print_metrics("Metrics on Threshold-Passing Samples", metrics_on_subset)
        print_metrics("Overall Metrics (All Data Processed by Model)", metrics_overall)
        print("\n")

    @staticmethod
    def report_pipeline_summary(all_model_reports, system_metrics):
        print("="*60)
        print("OVERALL SYSTEM PIPELINE SUMMARY")
        print("="*60)
        
        print("\n--- Final Pipeline Metrics ---")
        if not system_metrics:
            print("No system metrics available.")
            return

        print(f"Accuracy:        {system_metrics.get('accuracy', 0):.4f}")
        print(f"F1 (Weighted):   {system_metrics.get('f1_weighted', 0):.4f}")
        print(f"F1 (Macro):      {system_metrics.get('f1_macro', 0):.4f}")
        print(f"Precision:       {system_metrics.get('precision', 0):.4f}")
        print(f"Recall:          {system_metrics.get('recall', 0):.4f}")
        
        if system_metrics.get('roc_auc') is not None:
            print(f"ROC-AUC:         {system_metrics.get('roc_auc', 0):.4f}")

        if 'confusion_matrix' in system_metrics and 'label_order' in system_metrics:
            print("\nFinal System Confusion Matrix:")
            df_cm = pd.DataFrame(system_metrics['confusion_matrix'], 
                               index=system_metrics['label_order'], 
                               columns=system_metrics['label_order'])
            print(df_cm)
        print("\n")
