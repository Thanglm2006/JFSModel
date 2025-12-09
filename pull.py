from huggingface_hub import snapshot_download

# Define the model ID and where you want to save it
model_id = "Thanglm2006/JFS" # Replace with your model
local_folder1 = ("./models/step1")
local_folder2 = ("./models/step2")

print(f"Downloading {model_id} model1...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_folder1,
    allow_patterns=["step1/*"],
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
)

print(f"Downloading {model_id} model2...")

snapshot_download(
    repo_id=model_id,
    local_dir=local_folder2,
    allow_patterns=["step2/*"],
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Optional: Ignore duplicate/unused weights to save space
)

print("Download complete.")