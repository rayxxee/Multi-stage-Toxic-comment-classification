from locust import HttpUser, task, between
import random

class APIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_predict_single(self):
        sample_texts = [
            "This is a perfectly safe comment.",
            "You are such an idiot, I hate you.",
            "I disagree with your opinion, but I understand.",
            "What a beautiful day to write some code!",
            "Go kill yourself."
        ]
        
        payload = {
            "texts": [random.choice(sample_texts)]
        }
        
        self.client.post("/api/v1/predict", json=payload)

    @task(3)
    def test_predict_batch(self):
        sample_texts = [
            "This is a perfectly safe comment.",
            "You are such an idiot, I hate you.",
            "I disagree with your opinion, but I understand.",
            "What a beautiful day to write some code!",
            "Go kill yourself."
        ]
        
        # Select 2-5 random texts
        num_texts = random.randint(2, 5)
        texts = [random.choice(sample_texts) for _ in range(num_texts)]
        
        payload = {
            "texts": texts
        }
        
        self.client.post("/api/v1/predict", json=payload)
