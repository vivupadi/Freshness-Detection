# test_local.py
import onnxruntime as ort
import numpy as np
from PIL import Image

# Paths (update these to your local paths)
MODEL_PATH = "D:\\BI Team\\solution\\trained_model\\named-outputs\\model_output\\fruit_classifier.onnx"
#IMAGE_PATH = "D:\\BI Team\\archive\\dataset\\Test\\freshapples\\a_f002.png"  # Your local test image freshapple 
#IMAGE_PATH = "D:\\BI Team\\archive\\dataset\\Test\\rottenapples\\rotated_by_60_Screen Shot 2018-06-07 at 2.51.45 PM.png"  #mold
#IMAGE_PATH = "D:\\BI Team\\archive\\dataset\\Test\\freshapples\\rotated_by_15_Screen Shot 2018-06-08 at 5.04.59 PM.png"  #fresh but slight

IMAGE_PATH = "D:\\BI Team\\archive\\dataset\\Test\\rottenoranges\\rotated_by_15_Screen Shot 2018-06-12 at 11.34.13 PM.png" ##slight mold orange

print("Loading model...")
session = ort.InferenceSession(MODEL_PATH)

print("Loading image...")
image = Image.open(IMAGE_PATH).convert('RGB').resize((224, 224))

# Preprocess - ✅ FIXED: Ensure float32 throughout
img_array = np.array(image).astype(np.float32) / 255.0

# Normalize - ✅ FIXED: Use float32
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
img_array = (img_array - mean) / std

# HWC -> CHW -> NCHW - ✅ FIXED: Ensure float32
img_array = img_array.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

print(f"Input dtype: {img_array.dtype}")  # Should show float32
print(f"Input shape: {img_array.shape}")

print("Running inference...")
outputs = session.run(None, {'image': img_array})

# Get predictions
fruit_classes = ['apples', 'banana', 'bittergourd', 'capsicum', 
                'cucumber', 'okra', 'oranges', 'potato', 'tomato']

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Parse outputs
fruit_probs = softmax(outputs[0][0])
freshness_probs = softmax(outputs[1][0])
confidence_score = outputs[2][0][0]  # ✅ Confidence/moldness score

fruit_idx = np.argmax(fruit_probs)
freshness_idx = np.argmax(freshness_probs)

# ✅ Calculate moldness level (inverse of confidence)
moldness_level = 1 - confidence_score  # Higher = more moldy

print("\n" + "="*60)
print("PREDICTION RESULTS")
print("="*60)
print(f"Fruit Type:    {fruit_classes[fruit_idx].upper()}")
print(f"  Confidence:  {fruit_probs[fruit_idx]:.1%}")

print(f"\nFreshness:     {'FRESH' if freshness_idx == 0 else 'ROTTEN'}")
print(f"  Confidence:  {freshness_probs[freshness_idx]:.1%}")

# ✅ Display moldness
print(f"\nMoldness Level: {moldness_level:.1%}")
print(f"  Confidence Score: {confidence_score:.4f}")

# ✅ Moldness interpretation
if moldness_level < 0.2:
    status = "Very Fresh ✓"
elif moldness_level < 0.4:
    status = "Fresh"
elif moldness_level < 0.6:
    status = "Slightly Old"
elif moldness_level < 0.8:
    status = "Going Bad"
else:
    status = "Moldy/Rotten ✗"

print(f"  Status: {status}")

print("="*60)

# ✅ Detailed breakdown
print("\nDETAILED BREAKDOWN:")
print("-"*60)
print(f"Top 3 Fruit Predictions:")
top_3_idx = np.argsort(fruit_probs)[::-1][:3]
for i, idx in enumerate(top_3_idx, 1):
    print(f"  {i}. {fruit_classes[idx]:12s} - {fruit_probs[idx]:.1%}")

print(f"\nFreshness Probabilities:")
print(f"  Fresh:  {freshness_probs[0]:.1%}")
print(f"  Rotten: {freshness_probs[1]:.1%}")

print(f"\nQuality Metrics:")
print(f"  Confidence Score: {confidence_score:.4f} (higher = fresher)")
print(f"  Moldness Level:   {moldness_level:.4f} (higher = moldier)")
print("="*60)