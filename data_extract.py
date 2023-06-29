import os
import shutil
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Set the API credentials (replace with your own username and key)
api = KaggleApi()
api.authenticate()

# Set the competition name and the classes you want to download
competition_name = "siim-isic-melanoma-classification"
classes = ["benign","malignant"] #"benign" #malignant

# Create directories for each class
base_dir = "./melanoma_data"
os.makedirs(base_dir, exist_ok=True)
for class_name in classes:
    class_dir = os.path.join(base_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

# To Download all the images
# api.competition_download_files(competition_name, path=base_dir, quiet=False)
# shutil.unpack_archive(os.path.join(base_dir, competition_name + ".zip"), extract_dir=base_dir)

metadata_df = pd.read_csv(os.path.join(base_dir, "train.csv"))

# To download selected files
for class_name in classes:
    class_metadata = metadata_df[metadata_df["benign_malignant"] == class_name].head(175)
    image_ids = class_metadata["image_name"].values
    for image_id in image_ids:
        api.competition_download_file(competition_name, f"jpeg/train/{image_id}.jpg",
                                      path=os.path.join(base_dir, class_name))
        print(f"Downloaded image: {image_id}.jpg")

print("Download complete!")