import fasttext
import os
import tempfile
import numpy as np

class FastTextClassifier:
    def __init__(self, lr=0.5, epoch=25, wordNgrams=2, minn=3, maxn=6, dim=100, loss='softmax'):
        self.lr = lr
        self.epoch = epoch
        self.wordNgrams = wordNgrams
        self.minn = minn
        self.maxn = maxn
        self.dim = dim
        self.loss = loss
        self.model = None
        
        # 0: not_safe, 1: safe
        self.idx_to_str = {0: "__label__not_safe", 1: "__label__safe"}
        self.str_to_idx = {"__label__not_safe": 0, "__label__safe": 1}

    def train(self, texts, labels):
        print("Training FastText model...")
        # create temporary file for fasttext
        fd, temp_path = tempfile.mkstemp(suffix=".txt")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                for text, label in zip(texts, labels):
                    text = str(text).replace('\n', ' ')
                    label_str = self.idx_to_str.get(label, "__label__safe")
                    f.write(f"{label_str} {text}\n")
            
            self.model = fasttext.train_supervised(
                input=temp_path,
                lr=self.lr,
                epoch=self.epoch,
                wordNgrams=self.wordNgrams,
                minn=self.minn,
                maxn=self.maxn,
                dim=self.dim,
                loss=self.loss
            )
        finally:
            os.remove(temp_path)
        print("FastText trained.")

    def predict_with_threshold(self, texts, threshold=0.50):
        if not self.model:
            raise ValueError("Model not trained or loaded.")
            
        clean_texts = [str(t).replace('\n', ' ') for t in texts]
        # k=1 returns the top prediction for each text
        pred_labels, pred_probs = self.model.predict(clean_texts, k=1)
        
        labels_out = []
        probs_out = []
        passed_mask = []
        
        for pl, pp in zip(pred_labels, pred_probs):
            label_str = pl[0]
            prob = pp[0]
            label_idx = self.str_to_idx.get(label_str, 1) # Default safe
            
            labels_out.append(label_idx)
            probs_out.append(prob)
            passed = prob >= threshold
            passed_mask.append(passed)
            
        return np.array(labels_out), np.array(probs_out), np.array(passed_mask)

    def predict(self, texts):
        labels, _, _ = self.predict_with_threshold(texts, threshold=0.0)
        return labels
        
    def predict_proba(self, texts):
         clean_texts = [str(t).replace('\n', ' ') for t in texts]
         # Use k=1 because fasttext 'hs' loss returns identical duplicate probabilities for k>1
         pred_labels, pred_probs = self.model.predict(clean_texts, k=1)
         
         all_probs = []
         for pl, pp in zip(pred_labels, pred_probs):
             top_label_str = pl[0]
             p = pp[0]
             
             probs = [0.0, 0.0]
             idx = self.str_to_idx.get(top_label_str, 1) # Default safe
             
             # Calculate proper binary probabilities (sum to 1)
             probs[idx] = p
             probs[1 - idx] = 1.0 - p
             
             all_probs.append(probs)
             
         return np.array(all_probs)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)
        print(f"FastText saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        instance = cls()
        instance.model = fasttext.load_model(filepath)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            import tempfile, os
            fd, path = tempfile.mkstemp()
            try:
                os.close(fd)
                self.model.save_model(path)
                with open(path, 'rb') as f:
                    state['model_bytes'] = f.read()
            finally:
                os.remove(path)
            state['model'] = None
        return state

    def __setstate__(self, state):
        model_bytes = state.pop('model_bytes', None)
        self.__dict__.update(state)
        if model_bytes is not None:
            import tempfile, os, fasttext
            fd, path = tempfile.mkstemp()
            try:
                os.close(fd)
                with open(path, 'wb') as f:
                    f.write(model_bytes)
                self.model = fasttext.load_model(path)
            finally:
                os.remove(path)

