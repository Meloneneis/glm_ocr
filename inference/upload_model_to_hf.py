from huggingface_hub import HfApi

# 1. Initialize the API
api = HfApi()

# 2. Define your details
repo_id = "meloneneneis/glm_ocr_21jhd"  # Format: username/repo_name
folder_path = "./merged_21jhd"          # Path to your local folder

# 3. Create the repository (if it doesn't exist yet)
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# 4. Upload the folder
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Initial model upload"
)

print(f"Uploaded to https://huggingface.co/{repo_id}")