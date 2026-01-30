"""
Shared Fruit Classifier with built-in Prometheus drift monitoring
ONNX Runtime implementation for optimized CPU inference
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import time
import os
import threading
from pathlib import Path
from azure.storage.blob import BlobClient, ContainerClient
from prometheus_client import Counter, Histogram, Gauge

# ============================================
# Prometheus Metrics
# ============================================
prediction_counter = Counter(
    'fruit_predictions_total',
    'Total predictions',
    ['status']
)

prediction_latency = Histogram(
    'fruit_prediction_seconds',
    'Prediction latency',
    buckets=[0.1, 0.5, 1.0, 2.0]
)

model_load_time = Gauge(
    'model_load_seconds',
    'Model load time'
)

# Drift monitoring metrics
drift_test_accuracy = Gauge(
    'model_drift_test_accuracy',
    'Model accuracy on test data'
)

drift_test_samples = Gauge(
    'model_drift_test_samples_total',
    'Total test samples evaluated'
)

drift_last_check = Gauge(
    'model_drift_last_check_timestamp',
    'Last drift check timestamp (unix)'
)

class FruitClassifier:
    """Load ONNX model, make predictions, and monitor drift"""

    def __init__(self, blob_connection_string, container_name, blob_name):
        start_time = time.time()

        self.connection_string = blob_connection_string

        try:
            # Download ONNX model from Azure Blob
            self.blob_client = BlobClient.from_connection_string(
                conn_str=blob_connection_string,
                container_name=container_name,
                blob_name=blob_name
            )
            model_bytes = self.blob_client.download_blob().readall()

            # Create temporary file for ONNX model (onnxruntime needs file path) --since azure blob container --> virtual folder an dthen  onnx model.
            filename = os.path.basename(blob_name)
            self.model_path = f"/tmp/{filename}"
            with open(self.model_path, 'wb') as f:
                f.write(model_bytes)

            # Load ONNX model with CPU execution provider
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )

            # Get model input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]

            load_time = time.time() - start_time
            model_load_time.set(load_time)
            print(f"✓ ONNX Model loaded ({load_time:.2f}s)")
            print(f"  Input: {self.input_name}")
            print(f"  Outputs: {self.output_names}")

        except Exception as e:
            print(f"❌ Load failed: {str(e)}")
            raise

        self.class_names = ['apples', 'banana', 'bittergroud', 'capsicum', 'cucumber',
                           'okra', 'oranges', 'potato', 'tomato']

        # ImageNet normalization (used during training)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Drift monitoring settings
        self.baseline_accuracy = 0.90
        self.drift_threshold = 0.05

    def preprocess_image(self, image):
        """
        Preprocess PIL Image to ONNX model input format
        Returns: numpy array [1, 3, 224, 224]
        """
        # Resize to 224x224
        image = image.resize((224, 224), Image.BILINEAR)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image).astype(np.float32) / 255.0

        # Normalize with ImageNet mean/std
        img_array = (img_array - self.mean) / self.std

        # Convert HWC to CHW format
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension [1, 3, 224, 224]
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict_moldness_level(self, freshness_class, freshness_prob, confidence_score):
        """
        Estimate moldness level from classification + confidence

        Args:
            freshness_class: 0 (fresh) or 1 (rotten)
            freshness_prob: P(fresh) probability
            confidence_score: Regression output (0-1)

        Returns:
            moldness_level: String description
        """
        if freshness_class == 0:  # Predicted fresh
            if freshness_prob > 0.95 and confidence_score > 0.9:
                level = 'Very Fresh / Sehr Frisch'
            elif freshness_prob > 0.85 and confidence_score > 0.75:
                level = 'Fresh / Frisch'
            elif freshness_prob > 0.70:
                level = 'Good / Gut'
            else:
                level = 'OK / Geht'
        else:  # Predicted rotten
            # Use confidence_score (quality regression output)
            # Lower score = worse condition
            # Severity progression: Slight mold → Mold → Rotten (worst)

            if confidence_score < 0.25:
                # Very low score = worst condition (completely spoiled)
                level = 'Rotten / Verdorben'
            elif confidence_score < 0.4:
                # Low score = heavily moldy
                level = 'Mold / Schimmel'
            else:
                # Medium score = just starting to deteriorate
                level = 'Slight mold / Leichter Schimmel'

        return level

    def predict_from_bytes(self, image_bytes):
        """Predict from image bytes (for web service)"""
        start_time = time.time()

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.preprocess_image(image)

            # Run ONNX inference
            outputs = self.session.run(
                self.output_names,
                {self.input_name: image_tensor}
            )

            # Parse outputs (fruit_logits, freshness_logits, confidence)
            fruit_logits = outputs[0][0]  # [9]
            freshness_logits = outputs[1][0]  # [2]
            mold_confidence = outputs[2][0][0]  # scalar

            # Softmax for probabilities
            fruit_probs = self._softmax(fruit_logits)
            freshness_probs = self._softmax(freshness_logits)

            # Get predictions
            item_idx = np.argmax(fruit_probs)
            item_name = self.class_names[item_idx]
            item_conf = float(fruit_probs[item_idx])

            freshness_idx = np.argmax(freshness_probs)
            freshness_name = 'Fresh / Frisch' if freshness_idx == 0 else 'Rotten / Verdorben'
            freshness_conf = float(freshness_probs[freshness_idx])

            # Get mold level
            mold_level = self.predict_moldness_level(
                freshness_idx,
                freshness_probs[0],
                mold_confidence
            )

            latency = time.time() - start_time
            prediction_latency.observe(latency)
            prediction_counter.labels(status='success').inc()

            return {
                'item': item_name,
                'item_confidence': item_conf,
                'freshness': freshness_name,
                'freshness_confidence': freshness_conf,
                'mold': mold_level,
                'mold_confidence': mold_confidence
            }

        except Exception as e:
            prediction_counter.labels(status='error').inc()
            raise

    def predict_from_path(self, image_path):
        """Predict from image file path (for drift monitoring)"""
        start_time = time.time()

        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess_image(image)

            # Run ONNX inference
            outputs = self.session.run(
                self.output_names,
                {self.input_name: image_tensor}
            )

            # Parse outputs
            fruit_logits = outputs[0][0]
            freshness_logits = outputs[1][0]
            mold_confidence = outputs[2][0][0]

            # Softmax for probabilities
            fruit_probs = self._softmax(fruit_logits)
            freshness_probs = self._softmax(freshness_logits)

            # Get predictions
            item_idx = np.argmax(fruit_probs)
            item_name = self.class_names[item_idx]
            item_conf = float(fruit_probs[item_idx])

            freshness_idx = np.argmax(freshness_probs)
            freshness_name = 'Fresh' if freshness_idx == 0 else 'Rotten'

            # Get mold level (same as predict_from_bytes)
            mold_level = self.predict_moldness_level(
                freshness_idx,
                freshness_probs[0],
                mold_confidence
            )

            latency = time.time() - start_time
            prediction_latency.observe(latency)
            prediction_counter.labels(status='success').inc()

            return {
                'item': item_name,
                'item_confidence': item_conf,
                'freshness': freshness_name,
                'mold': mold_level
            }

        except Exception as e:
            prediction_counter.labels(status='error').inc()
            raise

    def _softmax(self, x):
        """Compute softmax values for array x"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()

    def check_drift(self, test_container, test_folder='test'):
        """Evaluate on test data and detect drift"""
        print("\n" + "="*60)
        print("📊 DRIFT CHECK")
        print("="*60)

        try:
            # Download test images from blob (returns list of (local_path, blob_name) tuples)
            test_images = self._download_test_data(test_container, test_folder)
            if not test_images:
                print("⚠️ No test data available")
                return None

            # Evaluate
            correct = 0
            total = 0

            print(f"\n🔍 Evaluating on {len(test_images)} samples...")

            for image_path, blob_name in test_images:
                try:
                    # Parse ground truth from blob path: test/freshapples/img.jpg
                    folder_name = Path(blob_name).parent.name.lower()

                    # Extract fruit type and freshness from folder name
                    if folder_name.startswith('fresh'):
                        true_freshness = 'Fresh'
                        true_fruit = folder_name[5:]  # Remove 'fresh' prefix
                    elif folder_name.startswith('rotten'):
                        true_freshness = 'Rotten'
                        true_fruit = folder_name[6:]  # Remove 'rotten' prefix
                    else:
                        continue  # Skip unknown folders

                    result = self.predict_from_path(image_path)
                    total += 1

                    # Check if prediction matches ground truth
                    pred_fruit = result['item'].lower()
                    pred_freshness = result['freshness']

                    if pred_fruit == true_fruit and pred_freshness == true_freshness:
                        correct += 1

                except Exception as e:
                    print(f"⚠️ Error: {str(e)}")
                    continue

            if total == 0:
                return None

            accuracy = correct / total

            # Check for drift
            drift_detected = (self.baseline_accuracy - accuracy) > self.drift_threshold

            # Update Prometheus metrics
            drift_test_accuracy.set(accuracy)
            drift_test_samples.set(total)
            drift_last_check.set(time.time())

            # Print results
            print(f"\nResults:")
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  Samples: {total}")
            print(f"  Baseline: {self.baseline_accuracy:.1%}")

            if drift_detected:
                print(f"\n⚠️ DRIFT DETECTED! Drop: {(self.baseline_accuracy - accuracy):.1%}")
            else:
                print(f"\n✓ No drift")

            print("="*60)

            return {
                'drift_detected': drift_detected,
                'accuracy': accuracy,
                'samples': total
            }

        except Exception as e:
            print(f"❌ Drift check failed: {str(e)}")
            return None

    def _download_test_data(self, container_name, test_folder='test', local_dir='./test_data_temp'):
        """Download test data from Azure blob

        Returns:
            List of tuples: (local_path, blob_name) for ground truth extraction
        """
        try:
            container_client = ContainerClient.from_connection_string(
                self.connection_string,
                container_name
            )

            Path(local_dir).mkdir(exist_ok=True)
            blobs = container_client.list_blobs(name_starts_with=test_folder)

            image_paths = []
            for blob in blobs:
                if blob.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    blob_client = container_client.get_blob_client(blob.name)
                    # Use unique filename to avoid collisions
                    safe_name = blob.name.replace('/', '_')
                    local_path = os.path.join(local_dir, safe_name)

                    with open(local_path, 'wb') as f:
                        f.write(blob_client.download_blob().readall())

                    # Return both local path and original blob name for ground truth
                    image_paths.append((local_path, blob.name))

            return image_paths

        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            return []

    def start_drift_monitoring(self, test_container, test_folder='test', interval_hours=24):
        """Start background drift monitoring every N hours"""
        def run_periodic_check():
            while True:
                try:
                    self.check_drift(test_container, test_folder)
                except Exception as e:
                    print(f"❌ Drift check error: {str(e)}")

                time.sleep(interval_hours * 3600)

        thread = threading.Thread(target=run_periodic_check, daemon=True)
        thread.start()
        print(f"✓ Drift monitoring started (every {interval_hours}h)")
        return thread
