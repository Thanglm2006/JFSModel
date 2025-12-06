from huggingface_hub import snapshot_download

# Define the model ID and where you want to save it
model_id = "Thanglm2006/JFS" # Replace with your model
local_folder = "./my_scam_model"

# Download the model
print(f"Downloading {model_id} to {local_folder}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_folder,
    local_dir_use_symlinks=False, # Essential: Ensures actual files are saved, not symlinks
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Optional: Ignore duplicate/unused weights to save space
)

print("Download complete.")