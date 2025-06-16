import requests
import os

input_dir = "test"  
output_dir = "output" 

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(input_dir, filename)
        print(f"Processing: {filename}")

        with open(file_path, "rb") as f:
            response = requests.post("http://localhost:5000/enhance", files={"image": f})
            if response.status_code == 200:
                result = response.json()
                output_path = result["output_path"]
                # Copy the file from server-saved path to local batch output folder
                saved_name = f"enhanced_{filename}"
                os.rename(output_path, os.path.join(output_dir, saved_name))
                print(f"Saved: {saved_name}")
            else:
                print(f"Failed: {filename} - {response.status_code}")
