from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import io, base64

# Create FastAPI app
app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class GenerateRequest(BaseModel):
    prompt: str

# Load Stable Diffusion model
print("⏳ Loading model... (this may take some time on first run)")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)
pipe = pipe.to("cpu")  # use "cuda" if GPU available
print("✅ Model loaded!")

@app.post("/generate")
async def generate_image(req: GenerateRequest):
    try:
        print(f"🎨 Generating image for prompt: {req.prompt}")
        image = pipe(req.prompt, num_inference_steps=50).images[0]

        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"message": "Image generated successfully!", "image": img_str}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Backend is running!"}
