from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import shutil
from sklearn.model_selection import train_test_split
import gdown
import zipfile


def process_data():

    # Define constants
    gdrive_download_url = "https://drive.google.com/file/d/1k990D3BM9leqwhyI0ZdFWOzCvcJQICOs/view?usp=sharing"

    dataset_zip_path = "/tmp/ASIdataset.zip"  # Temporary path for the downloaded ZIP file
    extracted_dataset_path = "/tmp/ASIdataset"  # Path where the dataset will be extracted
    output_dataset_dir = "/tmp/split_dataset"  # Output directory for the split dataset

    # Step 1: Download the dataset from Google Drive
    gdown.download(gdrive_download_url, dataset_zip_path, fuzzy=True, quiet=False)

    # Step 2: Unzip the dataset
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dataset_path)

    # Step 3: Split dataset into training and testing sets
    def split_dataset(base_dir, output_dir, test_size=0.3):

        # Define paths
        image_dir = os.path.join(base_dir, "ASIdataset", "images", "Images")
        annotation_dir = os.path.join(base_dir,"ASIdataset", "annotations", "Annotation")
        train_image_dir = os.path.join(output_dir, "train", "images")
        train_annotation_dir = os.path.join(output_dir, "train", "annotations")
        test_image_dir = os.path.join(output_dir, "test", "images")
        test_annotation_dir = os.path.join(output_dir, "test", "annotations")

        # Create output directories
        for dir_path in [train_image_dir, train_annotation_dir, test_image_dir, test_annotation_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Iterate over subdirectories (e.g., each breed folder)
        for folder_name in os.listdir(image_dir):
            image_subfolder = os.path.join(image_dir, folder_name)
            annotation_subfolder = os.path.join(annotation_dir, folder_name)

            if os.path.isdir(image_subfolder) and os.path.isdir(annotation_subfolder):
                images = sorted(os.listdir(image_subfolder))
                annotations = sorted(os.listdir(annotation_subfolder))

                # Ensure the number of images matches the number of annotations
                if len(images) != len(annotations):
                    print(
                        f"Mismatch in counts for {folder_name}: {len(images)} images, {len(annotations)} annotations.")
                    continue

                # Create full paths
                image_paths = [os.path.join(image_subfolder, img) for img in images]
                annotation_paths = [os.path.join(annotation_subfolder, ann) for ann in annotations]

                # Split data into train and test sets
                train_images, test_images, train_annotations, test_annotations = train_test_split(
                    image_paths, annotation_paths, test_size=test_size, random_state=42
                )

                # Copy files to output directories
                for img_path, ann_path in zip(train_images, train_annotations):
                    shutil.copy(img_path, os.path.join(train_image_dir, folder_name + "_" + os.path.basename(img_path)))
                    shutil.copy(ann_path,
                                os.path.join(train_annotation_dir, folder_name + "_" + os.path.basename(ann_path)))

                for img_path, ann_path in zip(test_images, test_annotations):
                    shutil.copy(img_path, os.path.join(test_image_dir, folder_name + "_" + os.path.basename(img_path)))
                    shutil.copy(ann_path,
                                os.path.join(test_annotation_dir, folder_name + "_" + os.path.basename(ann_path)))

                print(f"Processed folder: {folder_name}")

    # Call the split function
    split_dataset(base_dir=extracted_dataset_path, output_dir=output_dataset_dir, test_size=0.3)


# Define the DAG
with DAG(
        dag_id='data_processing_dag',
        start_date=datetime(2024, 10, 1),
        schedule_interval=None
) as dag:
    # Define the task
    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data
    )
