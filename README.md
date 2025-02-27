# MiniGPT

MiniGPT is a lightweight implementation of a GPT-style model, designed for quick experimentation and easy deployment on Hugging Face.

## Features
- Train and save a GPT-style model
- Upload model checkpoints and configuration to Hugging Face Hub
- Simple script for managing your repository

## Uploading to Hugging Face

To create a repository and upload your model:

```python
from huggingface_hub import HfApi, upload_file

repo_name = "miniGPT"  # Change this if needed
hf_username = "Santthosh"  # Your Hugging Face username

# Initialize API and create repo
api = HfApi()
api.create_repo(repo_id=f"{hf_username}/{repo_name}", exist_ok=True)

# Upload model and config
upload_file(path_or_fileobj=model_save_path, path_in_repo="miniGPT.pth", repo_id=f"{hf_username}/{repo_name}")
upload_file(path_or_fileobj=config_save_path, path_in_repo="config.json", repo_id=f"{hf_username}/{repo_name}")

print(f"Model uploaded to: https://huggingface.co/{hf_username}/{repo_name}")
```

## Usage

Once uploaded, you can load the model from Hugging Face:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("Santthosh/miniGPT")
```

## License
This project is licensed under the MIT License.
