import pandas as pd
from PIL import Image
import numpy as np

# Load the CSV data
data = pd.read_csv('path_to_your_file.csv')

# Add a new column to your DataFrame that contains the image names
data['image_name'] = data.index.astype(str) + '.jpg'

# Function to load and preprocess images
def load_images(data, folder):
    images = []
    for _, row in data.iterrows():
        img_name = row['image_name']
        img = Image.open(folder + img_name)
        img = img.resize((desired_width, desired_height))  # Resize image
        img = np.array(img) / 255.0  # Normalize pixel values
        images.append(img)
    return np.array(images)

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Load the images
train_images = load_images(train_data, 'path_to_train_images/')
val_images = load_images(val_data, 'path_to_val_images/')

# Extract labels for training and validation sets
train_labels = np.array(train_data['labels']) / label_scale_factor
val_labels = np.array(val_data['labels']) / label_scale_factor
