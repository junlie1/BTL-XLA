import firebase_admin
from firebase_admin import credentials, storage

#tạo Firebase
cred = credentials.Certificate("api-xla-nhom2-firebase-adminsdk-lbb9w-028e4aa5e4.json")
firebase_admin.initialize_app(cred, {"storageBucket": "api-xla-nhom2.appspot.com"})


def upload_image_to_firebase(image_path, filename):
    bucket = storage.bucket()

    # Định nghĩa đường dẫn trong Firebase
    firebase_path = f"images/{filename}"

    blob = bucket.blob(firebase_path)
    blob.upload_from_filename(image_path)

    # Tạo metadata với token truy cập bảo mật
    metadata = {
        "firebaseStorageDownloadTokens": "83b3d0ff-2092-4fed-bef7-f62c1d970990"  # Token tùy chỉnh giống ví dụ
    }
    blob.metadata = metadata
    blob.patch()  # Áp dụng metadata

    # Tạo đường dẫn URL với token
    url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{firebase_path.replace('/', '%2F')}?alt=media&token={metadata['firebaseStorageDownloadTokens']}"

    return url