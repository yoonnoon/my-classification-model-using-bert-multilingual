import logging
import os.path
import tarfile
import time

import boto3
import pandas


def load_data_from_csv(filepath):
    df = pandas.read_csv(filepath)
    return df.dropna()


def aws_client(service_name, region_name, access_key, secret_key, **opt):
    return boto3.client(
        service_name,
        region_name=region_name,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        **opt
    )


def upload_file_to_s3(target_file, object_name, bucket, access_key, secret_key, region, **opt):
    s3 = aws_client("s3", region, access_key, secret_key, **opt)
    with open(target_file, "rb") as f:
        s3.upload_fileobj(f, bucket, object_name)


def download_file_from_s3(target_file, object_name, bucket, access_key, secret_key, region, **opt):
    s3 = aws_client("s3", region, access_key, secret_key, **opt)
    with open(target_file, "wb") as f:
        s3.download_fileobj(bucket, object_name, f)


def make_tarfile(tar_filepath, source_dir):
    with tarfile.open(tar_filepath, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def extract_tarfile(extract_dir, tar_filepath):
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(extract_dir)


def prerequisites_check__state_is_true(state):
    try:
        assert state is True
    except AssertionError:
        raise RuntimeError("State must be TRUE!")


def aspect_logging(func):
    log = logging.getLogger()

    def wrap(*args, **kwargs):
        start = time.time()
        log.info("Start: %s." % func.__name__)
        ret = func(*args, **kwargs)
        end = time.time()
        log.info("End: %s, elapsed: %s." % (func.__name__, end - start))
        return ret

    return wrap
