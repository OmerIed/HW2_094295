import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ======================================================
# ======================================================
# ======================================================
# ======================================================

# This file is meant only for training and saving the model.
# You may use it for basic inspection of the model performance.
# Keep in mind that only the original file will be used to evaluate your performance.

# ======================================================
# ======================================================
# ======================================================
# ======================================================

np.random.seed(0)
torch.manual_seed(0)

print("Your working directory is: ", os.getcwd())

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 20
LR = 0.001

#base_dir = os.path.join("..", "data")
train_dir = os.path.join("", "hw2_094295", "data","train")
val_dir = os.path.join("", "val")

def load_datasets(train_dir, val_dir):
    """Loads and transforms the datasets."""
    # Resize the samples and transform them into tensors
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

    # Create a pytorch dataset from a directory of images
    train_dataset = datasets.ImageFolder(train_dir, data_transforms)
    val_dataset = datasets.ImageFolder(val_dir, data_transforms)

    return train_dataset, val_dataset

train_dataset, val_dataset = load_datasets(train_dir, val_dir)

class_names = train_dataset.classes
print("The classes are: ", class_names)

# Dataloaders initialization
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloaders = {'train': train_dataloader, 'val': val_dataloader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

NUM_CLASSES = len(class_names)

model_ft = models.resnet50(pretrained=False)

# Fit the last layer for our specific task
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()

model_ft = model_ft.to(device)
model_path = "hw2_094295/models/trained_model.pt"  # Replace with the path to your trained model

# Load the model state dictionary
state_dict = torch.load(model_path)

# Load the state dictionary into the model
model_ft.load_state_dict(state_dict)

# Set the model to evaluation mode
model_ft.eval()

running_loss = 0.0
running_corrects = 0
for inputs, labels in tqdm(dataloaders['val']):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
    # statistics
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
 
epoch_loss = running_loss / dataset_sizes['val']
epoch_acc = running_corrects.double() / dataset_sizes['val']
print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'val', epoch_loss, epoch_acc))
