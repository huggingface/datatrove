import os
from collections import deque
from fnmatch import fnmatch

import backoff as backoff


try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:

    class ClientError(Exception):
        pass


def _get_s3_path_components(s3_path):
    bucket_name, _, prefix = s3_path[len("s3://") :].replace("//", "/").partition(os.sep)
    return bucket_name, prefix


def s3_file_exists(cloud_path):
    """Checks if a given file to path exists. Currently only check for s3. If path is a folder path return False
    Args:
        cloud_path:

    Returns:
        bool
    """
    s3_client = boto3.client("s3")
    bucket_name, prefix = _get_s3_path_components(cloud_path)

    try:
        s3_client.head_object(Bucket=bucket_name, Key=prefix)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "404":
            raise exc
        return False


def s3_upload_file(local_path, cloud_path):
    """Args:
    local_path
    cloud_path
    """
    if not os.path.isfile(local_path):
        raise OSError(f"File {local_path} does not exist")
    bucket_name, prefix = _get_s3_path_components(cloud_path)

    # Upload the file
    s3_client = boto3.client("s3")
    s3_client.upload_file(local_path, bucket_name, prefix)


def s3_download_file(cloud_path, local_path):
    """Args:
    cloud_path
    local_path
    """
    bucket_name, prefix = _get_s3_path_components(cloud_path)
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)

    local_folder = os.path.dirname(local_path)
    os.makedirs(local_folder, exist_ok=True)
    bucket.download_file(prefix, local_path)


@backoff.on_exception(backoff.expo, ClientError, max_time=10 * 60, giveup=lambda e: "SlowDown" not in str(e))
def s3_get_file_stream(cloud_path):
    bucket_name, prefix = _get_s3_path_components(cloud_path)
    s3_client = boto3.client("s3")
    return s3_client.get_object(Bucket=bucket_name, Key=prefix)["Body"]


def _match_prefix(base_prefix, prefix, match_pattern=None):
    if not match_pattern:
        return True
    path_elements = os.path.relpath(prefix, base_prefix).split(os.path.sep)
    pattern_elements = os.path.normpath(match_pattern).split(os.path.sep)
    match_len = min(len(pattern_elements), len(path_elements))
    return fnmatch(os.path.sep.join(path_elements[:match_len]), os.path.sep.join(pattern_elements[:match_len]))


def s3_get_file_list(cloud_path, match_pattern=None, recursive=True):
    """Get list of relative paths to files in a utils folder with a given (optional) extension

    Args:
        cloud_path
        match_pattern
        recursive

    Returns:

    """
    s3_client = boto3.client("s3")
    bucket, main_prefix = _get_s3_path_components(cloud_path)

    paginator = s3_client.get_paginator("list_objects_v2")
    objects = []
    prefixes = deque()

    prefixes.append(main_prefix)
    while prefixes:
        prefix = prefixes.popleft()
        for resp in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            if recursive:
                prefixes.extend([next_prefix["Prefix"] for next_prefix in resp.get("CommonPrefixes", [])])
            objects.extend(
                [
                    os.path.relpath(x["Key"], main_prefix)
                    for x in resp.get("Contents", [])
                    if x["Key"] != prefix and _match_prefix(main_prefix, x["Key"], match_pattern)
                ]
            )
    return sorted(objects)
