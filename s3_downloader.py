import os
import boto3
from botocore.client import Config

def download_model_from_minio(bucket_name: str, prefix: str, local_dir: str):
    """
    Downloads a complete directory (model) from MinIO to a local cache directory.
    Returns True if downloaded/exists, False on error.
    """
    if not bucket_name:
        return False
        
    endpoint_url = os.getenv("MINIO_INTERNAL_ENDPOINT") or os.getenv("MINIO_EXTERNAL_ENDPOINT")
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")
    
    if not all([endpoint_url, access_key, secret_key]):
        print("[MinIO] Credentials missing. Skipping MinIO download.")
        return False

    prefix = prefix or ""
    if prefix.startswith('/'):
        prefix = prefix[1:]
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    # Basic heuristic: if config.json or gliner_config.json exists, assume model is fully downloaded
    if os.path.exists(os.path.join(local_dir, "config.json")) or os.path.exists(os.path.join(local_dir, "gliner_config.json")):
        print(f"[MinIO] Model already cached at {local_dir}")
        return True

    print(f"[MinIO] Downloading model from s3://{bucket_name}/{prefix} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    
    use_ssl = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        use_ssl=use_ssl,
        config=Config(signature_version='s3v4')
    )
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        file_count = 0
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                file_key = obj['Key']
                if file_key == prefix:
                    continue
                    
                relative_path = os.path.relpath(file_key, prefix) if prefix else file_key
                local_file_path = os.path.join(local_dir, relative_path)
                
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                print(f"  Downloading: {relative_path}...")
                s3_client.download_file(bucket_name, file_key, local_file_path)
                file_count += 1
                
        if file_count == 0:
            print(f"[MinIO] Error: No files found in s3://{bucket_name}/{prefix}")
            return False
            
        print("[MinIO] Download complete!")
        return True
    except Exception as e:
        print(f"[MinIO] Error downloading model: {e}")
        return False
