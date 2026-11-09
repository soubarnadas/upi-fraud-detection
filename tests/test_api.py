"""
Tests for FastAPI scoring endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from src.service.api import app


class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")

    def test_single_scoring(self):
        payload = {
            "step": 12,
            "type": "TRANSFER",
            "amount": 25000.0,
            "nameOrig": "C100234",
            "oldbalanceOrg": 30000.0,
            "nameDest": "C998123",
            "oldbalanceDest": 500.0,
            "device_type": "mobile"
        }
        res = self.client.post("/score", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("risk_score", data)
        self.assertTrue(0 <= data["risk_score"] <= 100)
        self.assertIn("risk_tier", data)
        self.assertIn("action", data)
        self.assertIn("top_shap_factors", data)
        self.assertLessEqual(len(data["top_shap_factors"]), 5)

    def test_batch_scoring(self):
        payload = {
            "transactions": [
                {
                    "step": 14,
                    "type": "TRANSFER",
                    "amount": 100.0,
                    "nameOrig": "C111",
                    "oldbalanceOrg": 5000.0,
                    "nameDest": "C222",
                    "oldbalanceDest": 2000.0
                },
                {
                    "step": 3,
                    "type": "CASH_OUT",
                    "amount": 150000.0,
                    "nameOrig": "C333",
                    "oldbalanceOrg": 150000.0,
                    "nameDest": "M444",
                    "oldbalanceDest": 0.0
                }
            ]
        }
        res = self.client.post("/batch_score", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 2)


if __name__ == '__main__':
    unittest.main()
