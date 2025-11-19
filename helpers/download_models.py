from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Shashura/gpt-2-hr-role",
    local_dir="models/gpt-2-hr-role",
    local_dir_use_symlinks=False,
)

print("Модель скачана в models/gpt-2-hr-role")
