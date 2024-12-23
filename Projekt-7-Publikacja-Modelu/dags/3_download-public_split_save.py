from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import shutil
import zipfile
from sklearn.model_selection import train_test_split
import kaggle
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def process_kaggle_data():


    kaggle_dataset = "jessicali9530/stanford-dogs-dataset"
    dataset_download_dir = "/tmp/kaggle_dataset"  # Temporary directory for Kaggle dataset
    extracted_dataset_path = os.path.join(dataset_download_dir, "ASIdataset2")
    output_dataset_dir = "/tmp/split_dataset"  # Directory for split dataset
    zip_output_path = "/tmp/split_dataset.zip"  # Final ZIP file path


    os.makedirs(dataset_download_dir, exist_ok=True)
    kaggle.api.dataset_download_files(kaggle_dataset, path=dataset_download_dir, unzip=True)


    def split_dataset(base_dir, output_dir, test_size=0.3):


        image_dir = os.path.join(base_dir, "images", "Images")
        annotation_dir = os.path.join(base_dir, "annotations", "Annotation")
        train_image_dir = os.path.join(output_dir, "train", "images")
        train_annotation_dir = os.path.join(output_dir, "train", "annotations")
        test_image_dir = os.path.join(output_dir, "test", "images")
        test_annotation_dir = os.path.join(output_dir, "test", "annotations")


        for dir_path in [train_image_dir, train_annotation_dir, test_image_dir, test_annotation_dir]:
            os.makedirs(dir_path, exist_ok=True)


        for folder_name in os.listdir(image_dir):
            image_subfolder = os.path.join(image_dir, folder_name)
            annotation_subfolder = os.path.join(annotation_dir, folder_name)

            if os.path.isdir(image_subfolder) and os.path.isdir(annotation_subfolder):
                images = sorted(os.listdir(image_subfolder))
                annotations = sorted(os.listdir(annotation_subfolder))


                if len(images) != len(annotations):
                    print(f"Mismatch in counts for {folder_name}: {len(images)} images, {len(annotations)} annotations.")
                    continue


                image_paths = [os.path.join(image_subfolder, img) for img in images]
                annotation_paths = [os.path.join(annotation_subfolder, ann) for ann in annotations]


                train_images, test_images, train_annotations, test_annotations = train_test_split(
                    image_paths, annotation_paths, test_size=test_size, random_state=42
                )


                for img_path, ann_path in zip(train_images, train_annotations):
                    shutil.copy(img_path, os.path.join(train_image_dir, folder_name + "_" + os.path.basename(img_path)))
                    shutil.copy(ann_path, os.path.join(train_annotation_dir, folder_name + "_" + os.path.basename(ann_path)))

                for img_path, ann_path in zip(test_images, test_annotations):
                    shutil.copy(img_path, os.path.join(test_image_dir, folder_name + "_" + os.path.basename(img_path)))
                    shutil.copy(ann_path, os.path.join(test_annotation_dir, folder_name + "_" + os.path.basename(ann_path)))

                print(f"Processed folder: {folder_name}")


    split_dataset(base_dir=extracted_dataset_path, output_dir=output_dataset_dir, test_size=0.3)


    with zipfile.ZipFile(zip_output_path, 'w') as zipf:
        for root, _, files in os.walk(output_dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dataset_dir)
                zipf.write(file_path, arcname)
                upload_to_google_drive(zip_output_path, "split_dataset.zip")

    print(f"Dataset split and zipped at {zip_output_path}")

def upload_to_google_drive(file_path, file_name):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Create and upload file
    file = drive.CreateFile({'title': file_name})
    file.SetContentFile(file_path)
    file.Upload()

    print(f"Uploaded {file_name} to Google Drive.")

# Define the DAG
with DAG(
    dag_id='kaggle_data_processing_dag',
    start_date=datetime(2024, 10, 1),
    schedule_interval=None
) as dag:
    # Define the task
    process_kaggle_data_task = PythonOperator(
        task_id='process_kaggle_data',
        python_callable=process_kaggle_data
    )
