from google.cloud import storage
import os

BUCKET = "gen-lang-client-0489066588-data"
SOURCE = "data"
DEST = "/app/data"

client = storage.Client()
bucket = client.bucket(BUCKET)

blobs = bucket.list_blobs(prefix=SOURCE)
for blob in blobs:
    dest_path = os.path.join(DEST, os.path.relpath(blob.name, SOURCE))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {blob.name} -> {dest_path}")
    blob.download_to_filename(dest_path)
print("All files downloaded successfully.")