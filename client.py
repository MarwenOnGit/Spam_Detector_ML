import requests
import json

# Endpoint of your local FastAPI server
API_URL = "http://127.0.0.1:8000"

def main():
    email_content = input("Enter email content: ")

    payload = {
        "message": email_content
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise error if request failed
        print("Prediction:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error communicating with the API:", e)

if __name__ == "__main__":
    main()
