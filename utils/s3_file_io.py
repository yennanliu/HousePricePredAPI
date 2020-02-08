import os, yaml
import pandas as pd
from boto3 import client
import boto3

class S3FileIO:
    """
    class load/dump s3 data
    """
    def __init__(self):
        with open('config/aws_creds.yml') as f:
            cconfig = yaml.load(f)
        s3_credentials =  config['s3']
        self.aws_access_key_id = s3_credentials['AWS_ACCESS_KEY_ID']
        self.aws_secret_access_key = s3_credentials['AWS_SECRET_ACCESS_KEY']
        self.region_name = s3_credentials['REGION_NAME']
        self.bucket_name = s3_credentials['BUCKET_NAME']
        self.conn = client('s3') 

    def _load_s3_file(self, bucket_name=self.bucket_name, s3file):
        s3 = boto3.resource('s3')
        obj = s3.Object(bucketname, s3file)
        body = obj.get()['Body'].read()
        print (body)
        # with open('scaler.pkl', 'wb') as data:
        #     s3.Bucket('bucket').download_fileobj(key, data)
        # try:
        #     s3.Bucket(BUCKET_NAME).download_file(s3file)
        # except botocore.exceptions.ClientError as e:
        #     if e.response['Error']['Code'] == "404":
        #         print("The object does not exist.")
        # else:
        #     raise

    def _get_s3_file_list(self, bucket_name=self.bucket_name):
        """
        load file list in s3 bucket
        bucket_name : suntory-data
        """
        for key in self.conn.list_objects(Bucket=bucket_name)['Contents']:
            print(key['Key'])

    def _load_s3_csv_as_df(self, bucket_name, file_name):
        """
        load and return s3 csv as dataframe
        
        :bucket_name : suntory-data
        :file_name   : filtered_10_vm_joined_data/transaction_10_vm_master_vm_master_product.csv
        """
        obj = self.conn.get_object(Bucket=bucket_name,Key=file_name)
        df = pd.read_csv(obj['Body'])
        return df