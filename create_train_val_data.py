from PIL import Image, ImageOps
import copy
import os
import time
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm
from pathlib import Path
import shutil
from glob import glob
import random
import math
# Get the current working directory
cwd = os.getcwd()

# Create a relative path using the cwd
CLEAN_DATA_FOLDER = os.path.join(cwd, "train_rotate_clean", "train")


def augment_images_random(label, transform, test_transform, aug_version, train_size=800, test_ratio=0.1, val_ratio = 0.2,
                            test_size=100, val_size=200):
    # Setup relevant folder paths
    input_folder_train = os.path.join(CLEAN_DATA_FOLDER, label)
    
    # Create temporary folder for input images specific label (with all train and val images)
    temp_folder = f'./data/2_augmented_{aug_version}/temp_{label}'
    Path(temp_folder).mkdir(parents=True, exist_ok=True)
    
    # Copy all clean images into temp folder
    input_folders = [input_folder_train]
    
    for input_folder in input_folders:
        for image in os.listdir(input_folder):
            # Copy the contents of source to destination
            dataset_type = input_folder.split('/')[-2]  # Train or valid set
            shutil.copy(f'{input_folder}/{image}', temp_folder)

    # Create output folder paths
    output_folder_train = f'./data/2_augmented_{aug_version}/train/{label}'
    output_folder_val = f'./data/2_augmented_{aug_version}/val/{label}'
    output_folder_test = f'./data/2_augmented_{aug_version}/test/{label}'

    Path(output_folder_train).mkdir(parents=True, exist_ok=True)
    Path(output_folder_val).mkdir(parents=True, exist_ok=True)
    Path(output_folder_test).mkdir(parents=True, exist_ok=True)

    input_files = glob(os.path.join(temp_folder, "*.png"))
    print(f'Created temp folder: {temp_folder}')
    existing_size = len(input_files)
    # Calculate the sizes of test, validation, and train sets based on ratios
    existing_test_size = math.ceil(existing_size * test_ratio)
    existing_val_size = math.ceil((existing_size - existing_test_size) * val_ratio)  # 20% of the remaining size for validation
    existing_train_size = existing_size - existing_test_size - existing_val_size  # The rest for training

    # Randomly shuffle the full image list
    random.shuffle(input_files)

    # Split the images into test, validation, and train sets
    test_list = input_files[:existing_test_size]
    remaining_images = input_files[existing_test_size:]

    val_list = remaining_images[:existing_val_size]
    train_list = remaining_images[existing_val_size:]
    
    # Copy all clean images into temp folder
    input_folders = [(train_list,output_folder_train),(val_list,output_folder_val),(test_list,output_folder_test)]
    
    for input_list,output_folder in input_folders:
        for image in input_list:
            # Copy the contents of source to destination 
            shutil.copy(f'{image}', output_folder)
            
    # Calculate number of additional images to generate for the wanted sizes
    train_balance_count = train_size - len(train_list)
    test_balance_count = test_size - len(test_list)
    val_balance_count = val_size - len(val_list)
    
    n = 0
    for i in range(train_balance_count):
        n += 1
        random_index = random.choice(range(len(train_list)))
        random_file = train_list[random_index]
        img_random = Image.open(random_file)
        img_random = ImageOps.grayscale(img_random)

        # Apply the transforms to the image
        transformed_image = transform(img_random)
            
        
        # Convert back to PIL to save
        transformed_image_pil = transforms.ToPILImage()(transformed_image)
     
        transformed_image_pil.save(os.path.join(output_folder_train,f'{label}_random_{n}.png'), 'PNG')
    
    n = 0
    for i in range(test_balance_count):
        n += 1
        random_index = random.choice(range(len(test_list)))
        random_file = test_list[random_index]
        img_random = Image.open(random_file)
        img_random = ImageOps.grayscale(img_random)

        # Apply the transforms to the image
        transformed_image = test_transform(img_random)
            
        
        # Convert back to PIL to save
        transformed_image_pil = transforms.ToPILImage()(transformed_image)
     
        transformed_image_pil.save(os.path.join(output_folder_test,f'{label}_random_{n}.png'), 'PNG')
    
    n = 0
    for i in range(val_balance_count):
        n += 1
        random_index = random.choice(range(len(val_list)))
        random_file = val_list[random_index]
        img_random = Image.open(random_file)
        img_random = ImageOps.grayscale(img_random)

        # Apply the transforms to the image
        transformed_image = transform(img_random)
            
        
        # Convert back to PIL to save
        transformed_image_pil = transforms.ToPILImage()(transformed_image)
     
        transformed_image_pil.save(os.path.join(output_folder_val,f'{label}_random_{n}.png'), 'PNG')
        
    # for file in train_list:
        # shutil.copy(os.path.join(temp_folder, file), output_folder_train)
    # for file in val_list:
        # shutil.copy(os.path.join(temp_folder, file), output_folder_val)
    # for file in test_list:
        # shutil.copy(os.path.join(temp_folder, file), output_folder_test)

    # Delete temporary folder
    shutil.rmtree(temp_folder, ignore_errors=True)
    

# Mixing train and valid sets together first, perform augmentation, and then splitting
def augment_images_shuffle(label, transform, aug_version, total_size=1000, train_size=800):

    # Setup relevant folder paths    
    input_folder_train = os.path.join(CLEAN_DATA_FOLDER,label)
    
    # Create temporary folder for input images specific label (with all train and val images)
    temp_folder = f'./data/2_augmented_{aug_version}/temp_{label}'
    Path(temp_folder).mkdir(parents=True, exist_ok=True)
   
    # Copy all clean images into temp folder
    input_folders = [input_folder_train]
    
    for input_folder in input_folders:
        for image in os.listdir(input_folder):
            # Copy the contents of source to destination 
            dataset_type = input_folder.split('/')[-2] # Train or valid set
            shutil.copy(f'{input_folder}/{image}', temp_folder)

    # Create output folder paths
    output_folder_train = f'./data/2_augmented_{aug_version}/train/{label}'
    output_folder_val = f'./data/2_augmented_{aug_version}/val/{label}'
    
    Path(output_folder_train).mkdir(parents=True, exist_ok=True)
    Path(output_folder_val).mkdir(parents=True, exist_ok=True)
        
    input_files = glob(os.path.join(temp_folder, "*.png"))
    print(f'Created temp folder: {temp_folder}')
    
    # For every image, do random transformations until hit the 1000 image mark (for each label)
    temp_folder_count = len(os.listdir(temp_folder))
    
    # Calculate number of additional images to generate to top up to 1000 images
    balance_count = total_size - temp_folder_count

    n = 0
    for i in range(balance_count):
        n += 1
        random_index = random.choice(range(len(input_files)))
        random_file = input_files[random_index]
        img_random = Image.open(random_file)
        img_random = ImageOps.grayscale(img_random)

        # Apply the transforms to the image
        transformed_image = transform(img_random)
            
        
        # Convert back to PIL to save
        transformed_image_pil = transforms.ToPILImage()(transformed_image)
     
        transformed_image_pil.save(f'{temp_folder}/{label}_random_{n}.png', 'PNG')
        
    # Random assign images into train and validation folders (for each label) for final split
    full_img_list = [file for file in os.listdir(temp_folder)]
    print(len(full_img_list))
 
    
    train_list = random.sample(full_img_list, train_size)
    val_list = [x for x in full_img_list if x not in train_list]

    for file in train_list:
        shutil.copy(os.path.join(temp_folder, file), output_folder_train)
    for file in val_list:
        shutil.copy(os.path.join(temp_folder, file), output_folder_val)
    # Delete temporary folder
    shutil.rmtree(temp_folder, ignore_errors=True)
    
 
# Define the TorchVision transforms for classic augmentations
transform_aug = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(5,sigma=(0.1,0.5)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomAdjustSharpness(1.1),
    #transforms.CenterCrop(180),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.ToTensor(),
])
                
      
target_labels = ['i','ii','iii','iv','v','vi','vii','viii','ix','x'] # List of all labels
for target_label in target_labels:
    augment_images_random(label=target_label, 
                            transform=transform_aug, 
                            test_transform=test_transform,
                            aug_version='v06',
                            train_size=900,
                            test_ratio=0.1,
                            val_ratio=0.1,
                            test_size=100,
                            val_size=100)   

exit()                
target_labels = ['i','ii','iii','iv','v','vi','vii','viii','ix','x'] # List of all labels
for target_label in target_labels:
    augment_images_shuffle(label=target_label, 
                            transform=transform_aug, # Indicate type of augmentation sequence
                            total_size=1000,
                            aug_version='v02', # Specify version number of experiment
                            train_size=800) # Creating a 80/20 train/val split                
