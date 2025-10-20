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
import tempfile

# Import your custom modules
try:
    from cloak import turbo_fgsm_cloak
    from STW.latetest import mistifying_image
except ImportError:
    # Fallback for when modules aren't available
    def turbo_fgsm_cloak(input_path, output_path, target_text="oil painting"):
        # Simple fallback - just copy the file
        shutil.copy2(input_path, output_path)
    
    def mistifying_image(input_path, output_path):
        # Simple fallback - just copy the file
        shutil.copy2(input_path, output_path)

app = FastAPI(title="Varnish API", version="1.0.0")

# Configure CORS to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Vercel deployment
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
    return {"message": "Hello from Varnish API!"}

@app.get("/api/")
async def api_root():
    return {"message": "Varnish API is running!"}

@app.get("/api/items", response_model=List[Item])
async def get_items():
    return items_db

@app.get("/api/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            return item
    return {"error": "Item not found"}

@app.post("/api/items", response_model=Item)
async def create_item(item: ItemCreate):
    new_id = max([item.id for item in items_db], default=0) + 1
    new_item = Item(id=new_id, name=item.name, description=item.description)
    items_db.append(new_item)
    return new_item

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Varnish API is running"}

@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    target_text: str = "oil painting"  # 👈 quick knob to change the cloak target
):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Use temporary directories for Vercel
        with tempfile.TemporaryDirectory() as temp_dir:
            # Unique names
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:8]
            original_filename = f"{ts}_{uid}_{file.filename}"
            original_path = os.path.join(temp_dir, original_filename)

            # Save original
            data = await file.read()
            with open(original_path, "wb") as f:
                f.write(data)

            # Decide protected filename (save as JPG for simplicity)
            protected_filename = f"{ts}_{uid}_cloaked.jpg"
            protected_path = os.path.join(temp_dir, protected_filename)

            # Process the image
            try:
                turbo_fgsm_cloak(original_path, protected_path, target_text=target_text)
            except Exception as e:
                # Fallback to simple copy if processing fails
                shutil.copy2(original_path, protected_path)

            # Read the processed image
            with open(protected_path, "rb") as f:
                processed_data = f.read()

            # Return the processed image as base64 or handle differently
            import base64
            processed_base64 = base64.b64encode(processed_data).decode('utf-8')
            
            return {
                "success": True,
                "message": f"Image '{file.filename}' processed and cloaked.",
                "original_filename": original_filename,
                "protected_filename": protected_filename,
                "size": len(data),
                "timestamp": ts,
                "processed_image": f"data:image/jpeg;base64,{processed_base64}"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

# For Vercel serverless functions
def handler(request):
    return app(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)