import json

from minio import Minio

from config.minio_config import minio_config
try:
    minio_client=Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=False
    )

    if not minio_client.bucket_exists(minio_config.bucket_name):
        minio_client.make_bucket(minio_config.bucket_name)

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}/*"]
                }
            ]
        }
        minio_client.set_bucket_policy(minio_config.bucket_name, json.dumps(policy))
except Exception as e:
    print(e)
    minio_client=None

def get_minio_client():
    return minio_client

if __name__ == "__main__":
    minio_client = get_minio_client()
    print(minio_client.bucket_exists("a123test"))