from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from datetime import datetime, timedelta, timezone
from utils.file_utils import get_secret
from urllib.parse import urlparse, unquote
import logging
import os
import time

class BlobStorageClient:

    def __init__(self, file_url):
        self.file_url = file_url

    def download_blob(self, file_url):
        parsed_url = urlparse(file_url)
        account_url = parsed_url.scheme + "://" + parsed_url.netloc
        container_name = parsed_url.path.split("/")[1]
        url_decoded = unquote(parsed_url.path)
        blob_name = url_decoded[len(container_name) + 2:]
        logging.info(f"[blob][{blob_name}] Connecting to blob.")

        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_error = None

        data = ""

        try:
            data = blob_client.download_blob().readall()
        except Exception as e:
            logging.info(f"[blob][{blob_name}] Connection error, retrying in 10 seconds...")
            time.sleep(10)
            try:
                data = blob_client.download_blob().readall()
            except Exception as e:
                blob_error = e

        if blob_error:
            error_message = f"Blob client error when reading from blob storage. {blob_error}"
            logging.info(f"[blob][{blob_name}] {error_message}")
        
        return data	


    def upload_blob(self, blob_name, data):
        parsed_url = urlparse(self.file_url)
        account_url = parsed_url.scheme + "://" + parsed_url.netloc
        container_name = f'{parsed_url.path.split("/")[1]}-tables' 

        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        # Create a blob client for the specific image
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_error = None

        try:
            # Upload the image to the blob, overwrite if the blob exists
            blob_client.upload_blob(data, blob_type="BlockBlob", overwrite=True)
            blob_url = f"{account_url}/{container_name}/{blob_name}"
        except Exception as e:
            logging.info(f"[blob][{blob_name}] Connection error, retrying in 10 seconds...")
            time.sleep(10)
            try:
                # Upload the image to the blob, overwrite if the blob exists
                blob_client.upload_blob(data, blob_type="BlockBlob", overwrite=True)
            except Exception as e:
                blob_error = e

        if blob_error:
            error_message = f"Blob client error when reading from blob storage. {blob_error}"
            logging.info(f"[blob][{blob_name}] {error_message}")

        logging.info(f"Uploaded {blob_name} to Azure Blob Storage")

        return blob_url
    

    def download_blob_locally(self, file_url):
        parsed_url = urlparse(file_url)
        account_url = parsed_url.scheme + "://" + parsed_url.netloc
        container_name = parsed_url.path.split("/")[1]
        url_decoded = unquote(parsed_url.path)
        blob_name = url_decoded[len(container_name) + 2:]
        logging.info(f"[blob][{blob_name}] Connecting to blob.")

        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_error = None

        local_file_path = f"/tmp/{blob_name}"

        try:
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        except Exception as e:
            logging.info(f"[blob][{blob_name}] Connection error, retrying in 10 seconds...")
            time.sleep(10)
            try:
                with open(local_file_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
            except Exception as e:
                blob_error = e

        if blob_error:
            error_message = f"Blob client error when reading from blob storage. {blob_error}"
            logging.info(f"[blob][{blob_name}] {error_message}")
        
        return local_file_path, blob_name
    

    def generate_sas_token(self, file_url):
        storage_account_name = os.environ.get("STORAGE_ACCOUNT_NAME")
        storage_key = get_secret("storage-account-key")
        parsed_url = urlparse(file_url)
        container_name = parsed_url.path.split("/")[1]
        url_decoded = unquote(parsed_url.path)
        blob_name = url_decoded[len(container_name) + 2:]
        connection_string = f"DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName={storage_account_name};AccountKey={storage_key}"
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        sas_token = ""
        try:
            sas_token = generate_blob_sas(
                account_name=blob_service_client.account_name,
                container_name=container_name,
                blob_name=blob_name,
                account_key=blob_service_client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(hours=1)  # Token valid for 1 hour
            )
        except Exception as e:
            blob_error = e
            error_message = f"Blob client error when generating SAS Token. {blob_error}"
            logging.info(f"[blob][{blob_name}] {error_message}")
        
        blob_url = f"{file_url}?{sas_token}"
        return blob_url