import requests

url = 'http://127.0.0.1:5000/api/chat'
payload = {'query': 'What are the details of Oorja Garg?'}

response = requests.post(url, json=payload)

print("Response Status Code:", response.status_code)
print("Response Text:", response.text)

try:
    print("Response JSON:", response.json())
except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON.")
