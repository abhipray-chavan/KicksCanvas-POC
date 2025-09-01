import csv
import re 
import os 

prompts_file = "prompts.txt"
image_folder = "./images/"
output_file = "dataset.csv"

dataset_data = []

print(f"reading prompts from the {prompts_file} file...")

with open(prompts_file, 'r') as f:
    for line in f:
        # Skip any empty lines
        if not line.strip():
            continue
        
        # Use a regular expression to find the number and the prompt text
        # This matches the "1. [prompt text]" format
        match = re.match(r'^(\d+)\.\s*(.*)', line)
        
        if match:
            number = match.group(1)
            prompt = match.group(2).strip()
            
            # Create the corresponding image filename (e.g., "1.jpg")
            image_filename = f"{number}.jpg"
            
            # Add the data to our list
            dataset_data.append([image_filename, prompt])


print(f"found {len(dataset_data)} prompts.")
print(f"writing dataset to {output_file}...")

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['image_filename', 'prompt'])
    writer.writerows(dataset_data)
    
print("Done! Your dataset.csv file has been created successfully.")