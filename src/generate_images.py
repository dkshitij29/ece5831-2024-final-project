import torch
import h5py
import numpy as np
from PIL import Image
from gan_factory import gan_factory  # Ensure the gan_factory creates the Generator
from txt2image_dataset import Text2ImageDataset  # Import the Dataset class
import os

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
GENERATOR_PATH = "./checkpoints/birds_results/birds/gen_190.pth"  # Update to your checkpoint path
DATASET_PATH = "./dataset/birds.hdf5"  # Path to the birds dataset
SAVE_PATH = "./generated_images"  # Directory to save images

# Create directory for saving images
os.makedirs(SAVE_PATH, exist_ok=True)

# Load the pretrained Generator
generator = gan_factory.generator_factory("gan").to(device)  # Create Generator
generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=device))
generator.eval()
print("Generator loaded successfully!")

# Load a batch of text embeddings from the dataset
with h5py.File(DATASET_PATH, "r") as dataset:
    # Extract sample text embeddings and descriptions
    text_embeddings = np.array(dataset["test"]["embeddings"][:8])  # Load 8 embeddings
    text_descriptions = dataset["test"]["txt"][:8]  # Corresponding text descriptions

# Convert embeddings to PyTorch tensors
text_embeddings = torch.FloatTensor(text_embeddings).to(device)

# Generate random noise
noise = torch.randn(text_embeddings.size(0), 100, 1, 1).to(device)

# Generate images
with torch.no_grad():
    fake_images = generator(text_embeddings, noise)

# Save the generated images
for i, img_tensor in enumerate(fake_images):
    img = img_tensor.cpu().detach().numpy()  # Convert to NumPy
    img = (img * 127.5 + 127.5).astype(np.uint8)  # Denormalize to [0, 255]
    img = np.transpose(img, (1, 2, 0))  # Change to HWC format

    # Save image
    image_path = os.path.join(SAVE_PATH, f"generated_image_{i}.jpg")
    Image.fromarray(img).save(image_path)
    print(f"Saved: {image_path}")

print("Image generation complete. Check the 'generated_images' folder.")
