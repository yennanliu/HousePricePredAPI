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
            config = yaml.load(f)
        self.aws_access_key_id = config['AWS_ACCESS_KEY_ID']
        self.aws_secret_access_key = config['AWS_SECRET_ACCESS_KEY']
        self.region_name = config['REGION_NAME']
        self.bucket_name = config['BUCKET_NAME']
        self.conn = client('s3') 

    def _load_s3_file(self, s3file):
        """
        load s3 file in memory
        : s3file :  the s3 file to be loaded in memory
        """
        s3 = boto3.resource('s3')
        try:
            obj = s3.Object(self.bucket_name, s3file)
            body = obj.get()['Body'].read()
            return body
        except Exception as e:
            print (">>> load s3 file failed!, s3file = {}".format(s3file))
            return

    def _download_s3_file(self, s3file, to_save_file):
        """
        dowload s3 file to local file system
        : s3file :  the s3 file to be downloaded 
        : to_save_file : the name of saved downloaded file
        """
        s3 = boto3.resource('s3')
        try:
            s3.meta.client.download_file(self.bucket_name, s3file, to_save_file)
            return 
        except Exception as e:
            print (">>> download s3 file failed!, s3file = {}, to_save_file = {}".format(s3file, to_save_file))

    def _upload_s3_file(self, s3file, to_upload_file):
        """
        upload file s3 cloud storage
        : s3file :  the name of uploaded file on s3
        : to_upload_file : the file to be uploaded to s3
        """
        s3 = boto3.resource('s3')
        try:
            s3.Bucket(self.bucket_name).upload_file(to_upload_file, s3file)
            return 
        except Exception as e:
            print (">>> download s3 file failed!, s3file = {}, to_upload_file = {}".format(s3file, to_upload_file))

    def _get_s3_file_list(self):
        """
        load file list in s3 bucket
        bucket_name : suntory-data
        """
        for key in self.conn.list_objects(Bucket=self.bucket_name)['Contents']:
            print(key['Key'])

    def _load_s3_csv_as_df(self, file_name):
        """
        load and return s3 csv as dataframe
        
        :bucket_name : suntory-data
        :file_name   : filtered_10_vm_joined_data/transaction_10_vm_master_vm_master_product.csv
        """
        obj = self.conn.get_object(Bucket=self.bucket_name,Key=file_name)
        df = pd.read_csv(obj['Body'])
        return df