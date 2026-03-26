#!/bin/bash
set -e

echo "Downloading embeddings..."
python download_embeddings.py
echo "Download complete, starting app..."

exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0