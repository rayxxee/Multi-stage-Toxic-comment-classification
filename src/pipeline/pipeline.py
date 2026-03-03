import numpy as np

class HierarchicalPipeline:
    def __init__(self, stages):
        """
        stages: list of dicts:
        [
            {'name': 'gatekeeper_lr', 'model': model_obj, 'threshold': 0.70, 'type': 'binary', 'pass_label': 2},
            ...
        ]
        """
        self.stages = stages
        
    def run(self, X, y_true=None):
        n_samples = len(X)
        y_final_pred = np.zeros(n_samples, dtype=int) - 1
        
        per_stage_data = {}
        active_indices = np.arange(n_samples)
        
        # Convert X to array for indexability
        X_array = np.array(X)
        y_true_array = np.array(y_true) if y_true is not None else None
        
        for i, stage in enumerate(self.stages):
            if len(active_indices) == 0:
                break
                
            model = stage['model']
            threshold = stage.get('threshold', 0.0)
            stage_name = stage['name']
            stage_type = stage.get('type', 'multiclass')
            pass_label = stage.get('pass_label', None)
            
            X_active = X_array[active_indices].tolist()
            
            # Probabilities for all active samples
            probs_full = model.predict_proba(X_active)
            labels_full = model.predict(X_active)
            
            if i < len(self.stages) - 1:
                # Not final stage, use threshold
                _, _, passed_mask = model.predict_with_threshold(X_active, threshold=threshold)
            else:
                # Final stage, everything passes
                passed_mask = np.ones(len(active_indices), dtype=bool)
                
            passed_indices = active_indices[passed_mask]
            
            if pass_label is not None and stage_type == 'binary':
                # If binary gatekeeper, all passing samples get the designated pass_label (e.g., 2 for Safe)
                passed_labels = np.full(len(passed_indices), pass_label)
                # Map full labels for reporting purposes
                labels_full_mapped = np.where(labels_full == 1, pass_label, 0)
            else:
                passed_labels = labels_full[passed_mask]
                labels_full_mapped = labels_full
            
            # Update final predictions mapping
            y_final_pred[passed_indices] = passed_labels
            
            # Save data for metrics
            stage_data = {
                'indices_processed': active_indices,
                'indices_passed': passed_indices,
                'y_pred_full': labels_full_mapped,
                'y_proba_full': probs_full,
                'y_pred_subset': passed_labels,
                'y_proba_subset': probs_full[passed_mask] if probs_full is not None else None
            }
            if y_true_array is not None:
                stage_data['y_true_full'] = y_true_array[active_indices]
                stage_data['y_true_subset'] = y_true_array[passed_indices]
                
            per_stage_data[stage_name] = stage_data
            
            # Move on with the rest
            active_indices = active_indices[~passed_mask]
            
        return y_final_pred, per_stage_data
