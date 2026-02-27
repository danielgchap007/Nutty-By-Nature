from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import os
import uuid

# Directory for image storage
IMAGE_UPLOADS_DIR = "image_uploads"

app = FastAPI()
app.mount("/images", StaticFiles(directory=IMAGE_UPLOADS_DIR), name="images")


# Simple server life check
@app.get("/")
def root():
    return {"status": "ok", "message": "FastAPI is running"}


# Upload images, save them to the image_uploads folder for model comparisons
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    # Ensure upload dir exists
    os.makedirs(IMAGE_UPLOADS_DIR, exist_ok=True)

    # Ensure filename exists
    if not file.filename:
        return {
            "response_status": "error",
            "error_msg": "No filename provided",
        }

    # Ensure a unique file path so that images will not
    # be overwritten
    extension = file.filename.split(".")[-1]
    unique_file_name = f"{uuid.uuid4()}.{extension}"

    # Create file path
    file_path = os.path.join(IMAGE_UPLOADS_DIR, unique_file_name)

    # Read file contents
    contents = await file.read()

    # Write to disk
    with open(file_path, "wb") as f:
        f.write(contents)

    return {
        "response_status": "success",
        "saved_as": file.filename,
        "content_type": file.content_type,
        "location": file_path,
    }
