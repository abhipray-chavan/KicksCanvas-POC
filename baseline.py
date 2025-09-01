import pandas as pd 
import torch 
from diffusers import StableDiffusionPipeline
import os 

# defining the paths and model id 
model_id = "runwayml/stable-diffusion-v1-5"
dataset_path = "dataset.csv"
output_path = "./output"
output_filename = "baseline_output.jpg"

# detecting the gpu , defaulting to cpu if not found . For windows system replace the line 37 with cuda 
# for mac users the below code should work 
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# loading the dataset
try :
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"Error: The file {dataset_path} was not found.")
    exit(1)

# selecting a prompt to use for baseline testing 
if not df.empty:
    prompt = df.iloc[0]['prompt']
else:
    print(f"Error: The dataset {dataset_path} is empty.")
    exit(1)

# loading the model 
print("Loading the model...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to(device)

# generating the image 
print("Generating the image...")
image = pipe(prompt).images[0]

# saving the image 
print("Saving the image...")
image.save(os.path.join(output_path, output_filename))

print("Done!")