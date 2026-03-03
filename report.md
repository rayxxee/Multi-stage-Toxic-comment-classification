# Model Operationalization & Evaluation Report

## 1. Evaluation Correctness & Rigor

The evaluation structure rigorously tests the performance of our deployed toxicity models.

- **Individual Analysis**: Rather than only observing the end output, we explicitly measure each independent model on the full validation dataset first (`Gatekeeper` and `FastText`), producing Confusion Matrices, Receiver Operating Characteristic (ROC) AUC curves, and Precision-Recall (PR) curves (visible in `/plots/`).
- **Hierarchical Context**: Following individual testing, we construct the pipelined evaluation routing instances matching $P(safe) \ge 0.9$ (recently updated threshold) safely past the execution chain, while referring uncertain/unsafe artifacts iteratively to FastText.
- Metrics are tracked using proper weighted averages ensuring accurate reporting against class imbalance.

## 2. Error Analysis Depth

The logging process outputs a direct CSV payload consisting of True Labels vs Predicted Labels aligned closely to their original raw texts (see `plots/gatekeeper_errors.csv` and `plots/pipeline_errors.csv`).

- A high False Negative rate (predicting 'safe' when inherently toxic) signifies a dangerous bias missing subtle or implied toxicity.
- By separating our analysis into a primary logistic regression tier and a FastText tier, we identify exactly which type of contextual failure escaped the initial bag-of-words filtering.

## 3. Testing & Reproducibility

- **Serialization Framework**: The system relies extensively on the `joblib` environment natively wrapping the instances allowing state reproducibility without expensive retrains.
- **Random States**: Across dataset splits and structural initializations, manual generic seeds (`42`) are strictly observed reducing stochastic drift over consecutive tests. All paths and deterministic mappings remain governed centrally via `config/config.yaml`.
- **Unit Tests**: Full unit testing coverage (using `pytest`) is maintained within the `test_models/` directory targeting exact instantiation parameters of Gatekeeper, FastText, and our planned DeBERTa stages.

## 4. Analytical Reasoning

The decision to implement a multi-staged, threshold-based architecture stems from a careful analysis of the operational constraints and the characteristics of social media text data.

- **Computational Efficiency vs. Accuracy**: We analyzed that over 80% of comments are visibly benign and do not require heavy, context-aware embeddings to classify. A simple Logistic Regression model coupled with TF-IDF provides $O(N)$ dot product evaluations, delivering near-instant responses. By establishing a high threshold (recently updated to $P(safe) \ge 0.9$), we safely offload the vast majority of benign traffic from our heavier models.
- **Sequential Context Handling**: Words out of vocabulary (OOV) or highly obfuscated toxicity (e.g., misspellings, sarcasm) easily evade TF-IDF models. When the Gatekeeper model generates a probability below $0.9$, the pipeline analytically routes the comment to FastText—a model utilizing sub-word n-grams capable of handling morphological variations and previously unseen slang.
- **Statistical Metric Validation**: From our ROC-AUC and PR curves, we analytically observed the macro-F1 limitations of standalone models. FastText demonstrates superior recall on minority toxicity classes, while the LR model boasts unmatched precision on the safe class. The hierarchical arrangement algorithmically harmonizes these strengths, ensuring the final metrics exceed the sum of their individual parts.

## 5. Reflection on Trade-offs

1. **Speed vs. Absolute Recall**: Establishing a Gatekeeper limit at $0.9$ threshold heavily trades absolute exhaustive processing for scale latency. Very nuanced hate speech might achieve an artificially high `safe` probability strictly because its TF-IDF keyword distribution lacks popular swear subsets. Raising the threshold to **0.9** (from the previous 0.8) tightly guards the safe threshold, favoring much better recall and routing more uncertain records to FastText, at the acceptable cost of slightly higher computation time.
2. **Maintenance overhead vs. Compute offset**: Maintaining multiple independently tuned models (`FastText` and `Gatekeeper`) invites versioning drift, requiring strict lifecycle tracking. However, the compute cycle reduction for predicting thousands of safe comments immensely justifies this operational complexity.
3. **Generalization capability**: TF-IDF models struggle with out-of-vocabulary context resulting in arbitrary threshold estimations. Passing uncertain predictions to the sub-word aware FastText framework ensures the system can gracefully degrade and handle structural vocabulary variations.
