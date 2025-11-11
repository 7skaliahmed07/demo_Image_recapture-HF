from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os

# Ultra-lightweight - no TensorFlow!
os.environ['OMP_NUM_THREADS'] = '1'  # Reduce memory usage

# Global session
session = None

app = FastAPI(title="Screen Recapture Detection")

@app.on_event("startup")
async def startup_event():
    global session
    print("🚀 Loading ONNX model...")
    try:
        # Use minimal session options
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        
        session = ort.InferenceSession(
            "model.onnx",
            sess_options=options,
            providers=['CPUExecutionProvider']  # CPU only
        )
        print("✅ ONNX model loaded successfully!")
        print(f"📊 Input: {session.get_inputs()[0].name}")
        print(f"📊 Output: {session.get_outputs()[0].name}")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

@app.get("/")
async def home():
    return """
    <html>
        <head><title>Screen Recapture Detection</title>
        <style>
            body { font-family: Arial; max-width: 500px; margin: 50px auto; padding: 20px; }
            .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
            .original { background: #d4edda; color: #155724; }
            .recaptured { background: #f8d7da; color: #721c24; }
        </style>
        </head>
        <body>
            <h1>📸 Screen Recapture Detection</h1>
            <p>Upload an image to check if it's original or recaptured</p>
            
            <div class="upload-form">
                <form id="uploadForm">
                    <input type="file" id="fileInput" accept="image/*" required>
                    <br><br>
                    <button type="submit">Analyze Image</button>
                </form>
            </div>
            
            <div id="result"></div>
            
            <script>
                document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    const file = document.getElementById('fileInput').files[0];
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = '<div class="result">Processing...</div>';
                    
                    try {
                        const response = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await response.json();
                        
                        if (data.success) {
                            const label = data.prediction.label;
                            const confidence = (data.prediction.confidence * 100).toFixed(2);
                            resultDiv.innerHTML = `
                                <div class="result ${label.includes('Original') ? 'original' : 'recaptured'}">
                                    <h3>${label}</h3>
                                    <p>Confidence: ${confidence}%</p>
                                </div>`;
                        } else {
                            resultDiv.innerHTML = `<div class="result">Error: ${data.error}</div>`;
                        }
                    } catch (error) {
                        resultDiv.innerHTML = `<div class="result">Error: ${error.message}</div>`;
                    }
                });
            </script>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global session
    
    if session is None:
        raise HTTPException(503, "Service starting...")
    
    try:
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run inference with ONNX
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        prediction = session.run([output_name], {input_name: img_array})[0][0][0]
        
        # Interpret results
        is_recaptured = prediction > 0.5
        label = "Recaptured Image" if is_recaptured else "Original Image"
        confidence = float(prediction if is_recaptured else 1 - prediction)
        
        return {
            "success": True,
            "prediction": {"label": label, "confidence": confidence, "raw_score": float(prediction)}
        }
        
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": session is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)