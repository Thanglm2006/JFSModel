from huggingface_hub import HfApi

api = HfApi()
REPO_ID = "Thanglm2006/JFS"

print("Dang upload Model 1...")
api.upload_folder(
    folder_path="./models/step1_mdeberta",  # Thư mục model 1 trên máy bạn
    repo_id=REPO_ID,
    repo_type="model",
    path_in_repo="step1"             # Tên thư mục con trên HF
)