import pandas as pd 
import torch 
import os 
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

# --- Configuration ---
dataset_path = "dataset.csv"
output_dir = "./controlNet_output_sdxl" 
control_map_dir = "./control_maps" 

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_id = "diffusers/controlnet-canny-sdxl-1.0" 

# --- NEW: Control panel for your POC ---
# Add the numbers of the specific images you want to generate.
# I've chosen 5 diverse examples to start with.
POC_IMAGE_NUMBERS = [1, 2, 3] 


# detecting the gpu 
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# loading the models and the pipeline 
print("Loading SDXL models...")
controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
)
pipe.to(device)

# loading the dataset
try :
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"Error: The file {dataset_path} was not found.")
    exit(1)

os.makedirs(output_dir, exist_ok=True)

# --- CHANGE: Filter the dataset to only include the selected POC images ---
df['base_number'] = df['image_filename'].str.split('.').str[0].astype(int)
poc_df = df[df['base_number'].isin(POC_IMAGE_NUMBERS)]

print(f"\nStarting POC generation for {len(poc_df)} selected images...")

for index, row in poc_df.iterrows():
    prompt = row['prompt']
    image_filename = row['image_filename']
    
    base_filename = os.path.splitext(image_filename)[0]
    controlmap_filename = f"{base_filename}_canny.png"
    full_controlmap_path = os.path.join(control_map_dir, controlmap_filename)
    
    if not os.path.exists(full_controlmap_path):
        print(f"Warning: Control map {full_controlmap_path} not found. Skipping.")
        continue
    
    print(f"Processing item #{base_filename}: {prompt[:50]}...")
    
    control_image = Image.open(full_controlmap_path)
    
    generated_image = pipe(
        prompt, 
        image=control_image,
        num_inference_steps=30
    ).images[0]
    
    output_filename = f"{base_filename}_sdxl_output.png"
    full_output_path = os.path.join(output_dir, output_filename)
    generated_image.save(full_output_path) 
    print(f"  -> Saved to {full_output_path}")
    
print("\nDone! Your POC image set has been generated.")