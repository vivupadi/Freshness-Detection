"""
FastAPI Fruit Freshness Classification Server 
Serves predictions with basic Prometheus monitoring
Uses ONNX Runtime for optimized CPU inference
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from prometheus_client import generate_latest


from src.inference.classifier import FruitClassifier

# ============================================
# Pydantic Models
# ============================================
class PredictionResponse(BaseModel):
    item: str
    item_confidence: float
    freshness: str
    freshness_confidence: float
    mold: str
    mold_confidence: float

# ============================================
# FastAPI Setup
# ============================================
app = FastAPI(title="Fruit Freshness Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = None

@app.on_event("startup")
async def startup_event():
    global classifier
    try:
        conn_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container = os.getenv("BLOB_CONTAINER")
        blob = os.getenv("BLOB_NAME")  # ONNX model file
        
        if not conn_string:
            print("⚠️ AZURE_STORAGE_CONNECTION_STRING not set")
            return
        
        classifier = FruitClassifier(conn_string, container, blob)

        # ✅ START 6-hour drift monitoring
        test_container = os.getenv("TEST_CONTAINER")
        test_folder = os.getenv("TEST_FOLDER")
        classifier.start_drift_monitoring(
            test_container=test_container,
            test_folder=test_folder,
            interval_hours=6
        )
        
        print("✓ App started with 6h drift monitoring")
        #print("✓ App started")
    except Exception as e:
        print(f"❌ Startup failed: {str(e)}")

# ============================================
# Routes
# ============================================
@app.get("/", response_class=FileResponse)
async def root():
    """Serve HTML UI"""
    return "frontend/index.html"

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Make prediction on image"""
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Must be image file")
    
    try:
        image_bytes = await file.read()
        result = classifier.predict_from_bytes(image_bytes)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics"""
    return generate_latest()

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }

@app.get("/info")
async def info():
    """Model info"""
    return {
        "name": "Fruit Freshness Classifier",
        "version": "1.0.0",
        "model": "ResNet50 Multi-task",
        "classes": ['apples', 'banana', 'bittergroud', 'capsicum', 'cucumber', 
                   'okra', 'oranges', 'potato', 'tomato'],
        "tasks": ["Fruit Classification", "Freshness Detection", "Mold Level Detection"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
