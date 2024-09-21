
import unittest
from fastapi.testclient import TestClient
from api.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

    def test_prediction_endpoint(self):
        test_data = {
            "data": "sample text for prediction",
            "model_type": "bert"
        }
        response = self.client.post("/predict", json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

if __name__ == '__main__':
    unittest.main()