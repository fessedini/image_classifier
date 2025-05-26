"""
batch_processing.py

This script processes incoming images in batches,
sends them to a Flask API for prediction,
and writes the results to a CSV file.
"""

import os
import requests
import pandas as pd

# Defin paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCOMING_IMAGES_PATH = os.path.join(BASE_DIR, "data", "incoming_images")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "predictions.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")

# Flask API endpoint
API_URL = 'http://127.0.0.1:5000/predict'

def process_images_in_batches(batch_size=32):
    """
    Processes incoming images in batches by sending them to a Flask API for prediction.

    Sends each image to the prediction endpoint, stores the predicted main and subcategory
    along with confidence scores, moves the processed images to a 'processed' folder,
    and saves the results to a CSV file.

    Args:
        batch_size (int): Number of images to process at once. Default is 32.
    """
    
    image_files = [f for f in os.listdir(INCOMING_IMAGES_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} new images to process.")
    
    if not image_files:
        print("No new images to process.")
        return
    
    results = []

    # Process in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        
        for file in batch_files:
            file_path = os.path.join(INCOMING_IMAGES_PATH, file)
            try:
                with open(file_path, 'rb') as img:
                    files = {'file': img}
                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        data = response.json()

                        main_category = data.get("predicted_main_category", "Unknown")
                        sub_category = data.get("predicted_subcategory", "Unknown")
                        main_probability = data.get("main_category_probability", 0)
                        sub_probability = data.get("subcategory_probability", 0)

                        # Save to results list
                        results.append({
                            'file_name': file,
                            'main_category': main_category,
                            'main_category_probability': main_probability,
                            'sub_category': sub_category,
                            'sub_category_probability': sub_probability
                        })

                        # Move the processed file to the processed folder
                        processed_path = os.path.join(PROCESSED_PATH, file)
                        os.rename(file_path, processed_path)
                    else:
                        error = response.text
                        try:
                            error = response.json()
                        except Exception:
                            pass
                        print(f"Failed to process {file}: {error}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Batch results saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    process_images_in_batches()