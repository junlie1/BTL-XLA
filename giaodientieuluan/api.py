import firebase_admin
from firebase_admin import credentials, storage

#tạo Firebase
cred = credentials.Certificate("api-xla-nhom2-firebase-adminsdk-lbb9w-7ca719683a.json")
firebase_admin.initialize_app(cred, {"storageBucket": "api-xla-nhom2.appspot.com"})


def upload_image_to_firebase(local_file_path, remote_file_path):
    bucket = storage.bucket()
    blob = bucket.blob(remote_file_path)
    blob.upload_from_filename(local_file_path)
    # Tạo URL công khai
    blob.make_public()
    return blob.public_url