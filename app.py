from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Force CPU usage for compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Lifespan context manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Loading Keras model...")
    app.state.model = tf.keras.models.load_model('final_model.keras')
    print("✅ Model loaded successfully!")
    yield
    # Shutdown (if needed)
    print("👋 Shutting down...")

app = FastAPI(title="Screen Recapture Detection", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple home page
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Screen Recapture Detection</title>
            <style>
                body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }
                .upload-form { border: 2px dashed #ccc; padding: 30px; text-align: center; margin: 20px 0; }
                button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
                .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
                .original { background: #d4edda; color: #155724; }
                .recaptured { background: #f8d7da; color: #721c24; }
            </style>
        </head>
        <body>
            <h1>📸 Screen Recapture Detection</h1>
            <p>Upload an image to detect if it's original or recaptured from a screen</p>
            
            <div class="upload-form">
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="fileInput" name="file" accept="image/*" required>
                    <br><br>
                    <button type="submit">Analyze Image</button>
                </form>
            </div>
            
            <div id="result"></div>
            
            <script>
                document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    const fileInput = document.getElementById('fileInput');
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    
                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        const resultDiv = document.getElementById('result');
                        
                        if (data.success) {
                            const label = data.prediction.label;
                            const confidence = (data.prediction.confidence * 100).toFixed(2);
                            
                            resultDiv.innerHTML = `
                                <div class="result ${label.includes('Original') ? 'original' : 'recaptured'}">
                                    <h3>${label}</h3>
                                    <p>Confidence: ${confidence}%</p>
                                    <p>Raw score: ${data.prediction.raw_score.toFixed(4)}</p>
                                </div>
                            `;
                        } else {
                            resultDiv.innerHTML = `<div class="result" style="background: #fff3cd; color: #856404;">
                                <p>Error: ${data.error}</p>
                            </div>`;
                        }
                    } catch (error) {
                        document.getElementById('result').innerHTML = `
                            <div class="result" style="background: #f8d7da; color: #721c24;">
                                <p>Network error: ${error.message}</p>
                            </div>`;
                    }
                });
            </script>
        </body>
    </html>
    """

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    
    try:
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess image (adjust size based on your model's requirements)
        img = img.resize((224, 224))  # Change this if your model expects different size
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = app.state.model.predict(img_array, verbose=0)[0][0]
        
        # Interpret results
        is_recaptured = prediction > 0.5
        label = "Recaptured Image" if is_recaptured else "Original Image"
        confidence = float(prediction if is_recaptured else 1 - prediction)
        
        return {
            "success": True,
            "prediction": {
                "label": label,
                "confidence": confidence,
                "raw_score": float(prediction)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Health check
@app.get("/health")
async def health():
    model_loaded = hasattr(app.state, 'model') and app.state.model is not None
    return {
        "status": "healthy", 
        "model_loaded": model_loaded,
        "message": "Screen Recapture Detection API is running!"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)