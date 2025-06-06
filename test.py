import requests

url = "https://sattriya-gesture-backend.onrender.com/predict"
file_path = r"C:\Projects\Temp\Testing\Dol\unscreen-076.png"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Failed to parse JSON:", e)
    print("Response Text:", response.text)
