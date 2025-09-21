from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import subprocess
from datetime import datetime
import uuid
import shutil


from cloak import turbo_fgsm_cloak
from STW.latetest import mistifying_image


app = FastAPI(title="Test API", version="1.0.0")

# Configure CORS to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class Item(BaseModel):
    id: int
    name: str
    description: str = None

class ItemCreate(BaseModel):
    name: str
    description: str = None

# In-memory storage for demo purposes
items_db = [
    Item(id=1, name="Test Item 1", description="This is a test item"),
    Item(id=2, name="Test Item 2", description="Another test item"),
]

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI backend!"}

@app.get("/items", response_model=List[Item])
async def get_items():
    return items_db

@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            return item
    return {"error": "Item not found"}

@app.post("/items", response_model=Item)
async def create_item(item: ItemCreate):
    new_id = max([item.id for item in items_db], default=0) + 1
    new_item = Item(id=new_id, name=item.name, description=item.description)
    items_db.append(new_item)
    return new_item

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    target_text: str = "oil painting"  # 👈 quick knob to change the cloak target
):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Ensure dirs
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("protected", exist_ok=True)

        # Unique names
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        original_filename = f"{ts}_{uid}_{file.filename}"
        original_path = os.path.join("uploads", original_filename)

        # Save original
        data = await file.read()
        with open(original_path, "wb") as f:
            f.write(data)

        # Decide protected filename (save as JPG for simplicity)
        protected_filename = f"{ts}_{uid}_cloaked.jpg"
        protected_path = os.path.join("protected", protected_filename)

        # NOTE: turbo_fgsm_cloak internally resizes to 224 for speed.
        mistifying_image(original_path, protected_path)
        # turbo_fgsm_cloak(original_path, protected_path, target_text=target_text)

        return {
            "success": True,
            "message": f"Image '{file.filename}' processed and cloaked.",
            "original_filename": original_filename,
            "protected_filename": protected_filename,
            "size": len(data),
            "timestamp": ts,
            "download_url": f"/download/{protected_filename}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Serve processed images"""
    file_path = os.path.join("protected", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='image/jpeg'
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
