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

## 6. Performance Testing & Analysis (Locust Stress Testing)

To ensure the API is robust under load, we conducted stress testing using **Locust** (`tests/locustfile.py`). We simulated 100 concurrent users spawning at a rate of 10 users per second, continuously hitting the `/predict` endpoint for 30 seconds.

**Empirical Results (30s test run):**
- **Total Requests:** 1,079
- **Throughput:** ~36 requests/second
- **Failure Rate:** 0%
- **Latency Percentiles:** Median (50%): 220 ms | 90%: 500 ms | 99%: 740 ms | Max: 970 ms

Analysis of the results shows:

- **Latency Analysis**: A median response time of 220ms is exceedingly fast for an ML sequential pipeline handling heavy concurrent traffic. The use of a pre-loaded, in-memory pipeline during FastAPI's `lifespan` event allows requests to experience minimal overhead. The Pydantic validation adds negligible latency, and the worst-case latency remained strictly under 1 second (970ms).
- **Throughput Profiling**: Processing ~36 requests per second flawlessly demonstrates FastAPI's asynchronous architecture handling concurrent I/O gracefully. Throughput is mainly bottlenecked by the GIL during CPU-bound model inference, but scaling worker processes (e.g., via `uvicorn src.api.main:app --workers 4`) allows linear scaling of request processing across multicore systems.
- **Stability Under Load**: Maintaining an exactly 0% failure rate proves the model inference and I/O queuing does not collapse under spike loading. Locust tasks included single texts and batches of 2-5 texts. Batch endpoints amortize the HTTP overhead, increasing total sentences processed per second significantly compared to singular requests.

## 7. System-Level Reasoning (Model as a Service)

The migration from an offline analytics script (`main.py`) to an online REST API (`src/api/main.py`) required careful system-level architectural decisions:

1. **Framework Choice**: **FastAPI** was chosen over Flask because of its native asynchronous support, autogenerated Swagger OpenAPI documentation, and deeply integrated **Pydantic** typing schemas. This guarantees incoming JSON payloads are structurally flawless before they reach the ML models.
2. **State Management**: ML models are huge objects. Loading `joblib` artifacts on every request is catastrophic for throughput. We load the `HierarchicalPipeline` strictly once synchronously via the `@asynccontextmanager lifespan`, attaching it to the `app.state`. This creates a transient, high-speed shared memory reference for all incoming HTTP requests.
3. **Robust Exits**: Utilizing `HTTPException` codes maps Python errors to correct REST primitives—returning `503 Service Unavailable` if models fail to load initially, and `422 Unprocessable Entity` for malformed input payloads.

## 8. Deployment Reflection

Taking models out of local memory and wrapping them in a network-facing service exposes distinct engineering realities:

- **Resource Provisioning**: During deployment, RAM becomes a vital constraint. Our Gatekeeper (LR) and FastText combinatorics must fit entirely in memory alongside the ASGI server overhead. Unloading unused objects or utilizing lighter quantized models becomes crucial in constrained CI/CD dockerized environments.
- **Fail-Safe Integrity**: If an individual pipeline model corrupts or fails `joblib` deserialization, the application gracefully degrades by refusing to start the prediction server while keeping the health endpoints alive, ensuring a load balancer would reroute traffic rather than returning corrupt outputs.
- **Scalability**: By containerizing this FastAPI application, the service can dynamically scale horizontally (HPA in Kubernetes). This perfectly encapsulates the ML pipeline, creating a true, robust Model-as-a-Service integration for varying frontend demands.
