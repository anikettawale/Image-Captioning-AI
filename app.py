from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

print("=" * 50)
print("      IMAGE CAPTIONING AI")
print("=" * 50)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nLoading AI Model... Please Wait...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

print("Model Loaded Successfully!")

while True:

    image_path = input("\nEnter Image Path : ")

    if not os.path.exists(image_path):
        print("Image not found!")
        continue

    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=30
    )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    print("\nGenerated Caption:")
    print("--------------------------------")
    print(caption)
    print("--------------------------------")

    again = input("\nTry another image? (y/n): ")

    if again.lower() != "y":
        print("\nThank You!")
        break
