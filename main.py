import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

# 1. Load your low-resolution image
input_path = r"C:\Users\bijan\Desktop\LunaX\lunar_samples\sample_14.png"
image = Image.open(input_path).convert("RGB")

# 2. Load pre-trained Swin2SR 4x upscaler model & processor
model_id = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
processor = AutoImageProcessor.from_pretrained(model_id)
model = Swin2SRForImageSuperResolution.from_pretrained(model_id)

# Move to GPU if CUDA is available for faster generation
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 3. Preprocess image & run inference
inputs = processor(image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# 4. Convert reconstructed tensor back to PIL Image
output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = (output * 255.0).round().astype("uint8").transpose(1, 2, 0)
high_res_image = Image.fromarray(output)

# 5. Save output image
high_res_image.save("ultra_high_res_output.jpg")
print("Image upscaled successfully!")