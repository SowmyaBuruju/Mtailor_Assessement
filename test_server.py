# test_server.py
import requests
import argparse
import os

def test_remote_inference(api_key, endpoint_url, image_path):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    files = {
        "file": open(image_path, "rb")
    }
    response = requests.post(endpoint_url, headers=headers, files=files)
    try:
        result = response.json()
    except Exception as e:
        print("Invalid response", response.text)
        return

    if response.status_code == 200:
        print(f"Prediction: Class ID = {result['class_id']}")
    else:
        print(f"Error: {result}")

def run_tests(api_key, endpoint_url):
    for img, label in [("n01440764_tench.jpeg", 0), ("n01667114_mud_turtle.JPEG", 35)]:
        print(f"\nTesting {img}...")
        test_remote_inference(api_key, endpoint_url, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Cerebrium API Key")
    parser.add_argument("--endpoint", required=True, help="Cerebrium Model Endpoint")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--run-tests", action="store_true", help="Run all predefined image tests")
    args = parser.parse_args()

    if args.run_tests:
        run_tests(args.api_key, args.endpoint)
    elif args.image:
        test_remote_inference(args.api_key, args.endpoint, args.image)
    else:
        print("Specify either --image or --run-tests")
