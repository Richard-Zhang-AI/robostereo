import json
import os
from pathlib import Path
from huggingface_hub import create_repo, upload_folder, login

root = Path("/nfs/rczhang/code/RoboStereo/tmp")
repo_id = "Richard-ZZZZZ/RoboStereo"
allowed_groups = {"oepl", "openvla", "assets"}

token = os.environ["HF_TOKEN"]

login(token=token, add_to_git_credential=True)
create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

manifest = json.loads((root / "hf_artifacts_manifest.json").read_text(encoding="utf-8"))
entries = [e for e in manifest["entries"] if e["group"] in allowed_groups]

for entry in entries:
    folder_path = root / entry["folder_path"]
    if not folder_path.is_dir():
        print(f"Skip missing: {folder_path}")
        continue
    print(f"Uploading {entry['folder_path']} ...")
    upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        path_in_repo=entry["folder_path"],
        token=token,
    )

print("All large-data entries uploaded.")