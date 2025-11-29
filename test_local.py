import requests
import json

# The URL of your locally running API
api_url = "http://localhost:8000/extract-bill-data"

# YOUR LOCAL FILE PATH (Use 'r' before the string to handle backslashes correctly)
file_path = r"C:\Users\LIKITH SALLA\OneDrive\Documents\train_sample_15.pdf"

payload = {
    "document": file_path
}

print(f"Sending request for file: {file_path}")
print("Waiting for Gemini to process...")

try:
    response = requests.post(api_url, json=payload)
    
    if response.status_code == 200:
        print("\n--- SUCCESS! ---")
        # Pretty print the JSON response
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"\n--- FAILED (Status {response.status_code}) ---")
        print(response.text)

except Exception as e:
    print(f"Error connecting to server: {e}")
    print("Make sure 'python main.py' is running in another terminal!")