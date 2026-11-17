import io
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


# Create FastAPI application
app = FastAPI(
    title="VisionPort Gateway (Session 2 prototype)"
)


# Select CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load YOLOv8 model
model = YOLO("yolov8n.pt")


# Home Endpoint
@app.get("/")
def home():
    return {
        "project": "VisionPort",
        "message": "VisionPort FastAPI Gateway is running!",
        "status": "success",
        "device": device,
        "available_endpoints": [
            "/detect",
            "/docs"
        ]
    }


# Object Detection Endpoint
@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = 0.35
):

    # Validate confidence
    if not 0 <= conf <= 1:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1"
        )

    # Validate image format
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=415,
            detail="Send a JPEG or PNG image"
        )

    # Read uploaded image
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty"
            )

        print("Filename:", file.filename)
        print("File size:", len(contents), "bytes")

        # Decode image
        image = Image.open(io.BytesIO(contents))
        image.load()
        image = image.convert("RGB")

        print("Image decoded successfully!")

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError, ValueError) as e:
        print("Image decoding error:", repr(e))

        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image: {str(e)}"
        )

    finally:
        await file.close()

    # Perform YOLO prediction
    try:
        r = model.predict(
            image,
            conf=conf,
            device=device,
            verbose=False
        )[0]

    except Exception as e:
        print("YOLO prediction error:", repr(e))

        raise HTTPException(
            status_code=500,
            detail="Object detection failed"
        )

    # Extract detected objects
    detections = [
        {
            "class": r.names[int(c)],
            "confidence": round(float(s), 3),
            "box_xyxy": [
                round(float(v), 1) for v in b
            ]
        }
        for c, s, b in zip(
            r.boxes.cls,
            r.boxes.conf,
            r.boxes.xyxy
        )
    ]

    # Return JSON response
    return {
        "image_size": {
            "width": image.width,
            "height": image.height
        },

        "num_detections": len(detections),

        "detections": detections,

        "timing_ms": {
            k: round(v, 2)
            for k, v in r.speed.items()
        }
    }