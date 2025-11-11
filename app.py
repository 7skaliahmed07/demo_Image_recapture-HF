from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gc

# MAXIMUM MEMORY OPTIMIZATION
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress ALL TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # No GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

# Reduce TensorFlow memory footprint
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = FastAPI(title="Screen Recapture Detection")

# Load model at module level (not in startup)
print("🚀 PRE-LOADING TensorFlow Lite model...")
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model pre-loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    interpreter = None

@app.get("/")
async def home():
    return """
    <html>
        <head><title>Screen Recapture Detection</title>
        <style>
            body { font-family: Arial; max-width: 500px; margin: 0 auto; padding: 20px; }
            .container { text-align: center; }
            .upload-form { border: 2px dashed #ccc; padding: 20px; margin: 20px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; margin: 10px; }
            .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
            .original { background: #d4edda; color: #155724; }
            .recaptured { background: #f8d7da; color: #721c24; }
            .error { background: #fff3cd; color: #856404; }
            .warning { background: #e2e3e5; color: #383d41; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
        </head>
        <body>
            <div class="container">
                <div class="warning">
                    <strong>Note:</strong> Free service may be slow to start. If error occurs, wait 30 seconds and try again.
                </div>
                
                <h1>📸 Screen Recapture Detection</h1>
                <p>Upload an image to check if it's original or recaptured from a screen</p>
                
                <div class="upload-form">
                    <form id="uploadForm">
                        <input type="file" id="fileInput" accept="image/*" required>
                        <br><br>
                        <button type="submit">Analyze Image</button>
                    </form>
                </div>
                
                <div id="result"></div>
            </div>
            
            <script>
                let isProcessing = false;
                
                document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    if (isProcessing) return;
                    isProcessing = true;
                    
                    const file = document.getElementById('fileInput').files[0];
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const resultDiv = document.getElementById('result');
                    const submitBtn = document.querySelector('button[type="submit"]');
                    
                    submitBtn.disabled = true;
                    submitBtn.textContent = 'Processing...';
                    resultDiv.innerHTML = '<div class="result">🔄 Analyzing image... Please wait.</div>';
                    
                    try {
                        const response = await fetch('/predict', { 
                            method: 'POST', 
                            body: formData,
                            headers: {
                                'Accept': 'application/json'
                            }
                        });
                        
                        if (!response.ok) {
                            throw new Error(`Server error: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        
                        if (data.success) {
                            const label = data.prediction.label;
                            const confidence = (data.prediction.confidence * 100).toFixed(2);
                            resultDiv.innerHTML = `
                                <div class="result ${label.includes('Original') ? 'original' : 'recaptured'}">
                                    <h3>${label}</h3>
                                    <p>Confidence: ${confidence}%</p>
                                    <p><small>Raw score: ${data.prediction.raw_score.toFixed(4)}</small></p>
                                </div>`;
                        } else {
                            resultDiv.innerHTML = `<div class="result error">Error: ${data.error}</div>`;
                        }
                    } catch (error) {
                        resultDiv.innerHTML = `
                            <div class="result error">
                                <p>Error: ${error.message}</p>
                                <p><small>This might be a temporary issue. Please try again in 30 seconds.</small></p>
                            </div>`;
                    } finally {
                        submitBtn.disabled = false;
                        submitBtn.textContent = 'Analyze Image';
                        isProcessing = false;
                    }
                });
            </script>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global interpreter
    
    if interpreter is None:
        raise HTTPException(503, "Model not loaded. Service may be restarting due to memory limits. Please try again in 30 seconds.")
    
    # Validate file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, "Please upload a valid image file (JPEG, PNG, etc.)")
    
    try:
        # Read with size limit
        contents = await file.read()
        if len(contents) > 2 * 1024 * 1024:  # 2MB limit
            raise HTTPException(400, "Image too large (max 2MB)")
        
        # Process image
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((128, 128))  # Smaller size for less memory
        
        # Convert to array
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run inference
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0][0]
        
        # FORCE MEMORY CLEANUP
        del img_array, img, contents
        gc.collect()
        
        # Return result
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
        
    except HTTPException:
        raise
    except Exception as e:
        # Force cleanup on error
        gc.collect()
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.get("/health")
async def health():
    return {
        "status": "healthy" if interpreter is not None else "restarting",
        "model_loaded": interpreter is not None,
        "memory_optimized": True
    }

@app.get("/simple")
async def simple_test():
    """Simple test endpoint without image processing"""
    return {"status": "ok", "model_loaded": interpreter is not None}

if __name__ == "__main__":
    import uvicorn
    # Single worker, no reload, minimal config
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        workers=1,
        log_level="error"
    )