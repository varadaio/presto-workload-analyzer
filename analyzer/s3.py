import boto3
import logbook

log = logbook.Logger("s3")


class S3:
    client = None
    s3_default_bucket_name = "rzp-edh-dev"

    def _init_client(self):
        # Setup a connection to S3
        self.client = boto3.client('s3')

    def get_client(self):
        if self.client is None:
            # Initialize client
            self._init_client()

        return self.client

    def upload(self, local_file_path, file_key, bucket_name=s3_default_bucket_name, extra_args=None):
        if not extra_args:
            extra_args = {'ACL': 'bucket-owner-full-control'}
        try:
            log.info(f"Uploading file to path: {file_key}")

            self.get_client().upload_file(
                local_file_path,
                bucket_name,
                file_key,
                ExtraArgs=extra_args
            )
        except Exception as e:
            log.error(e, file_key)
            raise


s3_provider = S3()
