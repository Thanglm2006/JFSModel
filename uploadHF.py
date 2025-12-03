from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="./my_scam_model",
    repo_id="Thanglm2006/JFS",
    repo_type="model"
)