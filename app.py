from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gc

# MEMORY OPTIMIZATION: Use TensorFlow Lite instead of full TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Global variables
interpreter = None
input_details = None
output_details = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - load TensorFlow Lite model
    global interpreter, input_details, output_details
    
    print("🚀 Loading TensorFlow Lite model...")
    
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ TensorFlow Lite model loaded successfully!")
        print(f"📊 Input shape: {input_details[0]['shape']}")
        print(f"📊 Input type: {input_details[0]['dtype']}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")
    gc.collect()

app = FastAPI(title="Screen Recapture Detection - TFLite", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                .info { background: #d1ecf1; color: #0c5460; padding: 10px; border-radius: 5px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="info">
                <strong>🚀 Memory Optimized Version</strong> - Using TensorFlow Lite for better performance
            </div>
            
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
                    
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = '<div class="result">🔄 Processing image... Please wait.</div>';
                    
                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            const errorText = await response.text();
                            throw new Error(`Server error: ${response.status} - ${errorText}`);
                        }
                        
                        const data = await response.json();
                        
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
                        resultDiv.innerHTML = `
                            <div class="result" style="background: #f8d7da; color: #721c24;">
                                <p>Error: ${error.message}</p>
                                <p><small>If this persists, the service might be restarting. Try again in 30 seconds.</small></p>
                            </div>`;
                    }
                });
            </script>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global interpreter, input_details, output_details
    
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service may be initializing.")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    
    try:
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess image - adjust size based on your model's expected input
        # Check what size your model expects and use that
        target_size = (224, 224)  # Change this if your model expects different size
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction results
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
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

@app.get("/health")
async def health():
    global interpreter
    return {
        "status": "healthy" if interpreter is not None else "initializing",
        "model_loaded": interpreter is not None,
        "message": "Screen Recapture Detection API (TensorFlow Lite)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1)