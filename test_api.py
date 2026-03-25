import requests

response = requests.post("http://127.0.0.1:8000/predict", json={
    "Age": 25,
    "Annual Income (k$)": 60,
    "Spending Score (1-100)": 70,
    "Gender_Female": 0,
    "Gender_Male": 1
})

print("Status:", response.status_code)
print("Response:", response.json())