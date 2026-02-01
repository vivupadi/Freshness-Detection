"""
For single images
Uses ONNX Runtime for optimized CPU inference
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

sys.path.insert(0, './src')
from src.inference.classifier import FruitClassifier

def main():
    # Load environment variables from .env file
    load_dotenv()


    image_path = sys.argv[1]

    # Verify image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        sys.exit(1)

    # Get configuration from environment (same as app.py)
    conn_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("BLOB_CONTAINER")
    blob = os.getenv("BLOB_NAME")

    if not conn_string:
        print("❌ Error: AZURE_STORAGE_CONNECTION_STRING not set in .env file")
        sys.exit(1)

    print(f"\n🔧 Configuration:")
    print(f"  Container: {container}")
    print(f"  Model: {blob}")
    print(f"  Image: {image_path}\n")

    # Load model
    print("Loading model...")
    classifier = FruitClassifier(conn_string, container, blob)

    # Predict
    print(f"\n🔍 Analyzing image: {image_path}")
    result = classifier.predict_from_path(image_path)

    # Display results
    print("\n" + "="*50)
    print("🎯 PREDICTION RESULTS")
    print("="*50)
    print(f"Item:              {result['item'].upper()}")
    print(f"  Confidence:      {result['item_confidence']:.1%}")
    print(f"\nFreshness:         {result['freshness']}")
    print(f"\nMold Level:        {result['mold']}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
