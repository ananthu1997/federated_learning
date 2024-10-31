#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import torch
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import random_split, DataLoader, Dataset 
from PIL import Image
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import cnn_iid

class CustomImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
def preprocess_images(dir, resize=(256,256)):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    transform_1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),

    ])
    transform_2 = transforms.Compose([
         transforms.Resize(resize),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    images = []
    labels = []
    label_counts = {"Normal": 0, "Benign": 0, "Malignant": 0}
    cancer_annot_str = {0: "Normal", 1: "Benign", 2: "Malignant"}
    folder_name = os.path.basename(os.path.normpath(dir))
    print(folder_name)
    for filename in os.listdir(dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(dir, filename)
            image = Image.open(image_path)
            img=image
            image = transform(image)
            images.append(image)
            label, cancer_str = label_names(filename)
            labels.append(label)
            label_counts[cancer_annot_str[label]] +=1
            augmented_image = img
            if label == 1:
                for _ in range(4):
                    augmented_image = transform_1(augmented_image)
                    aug_img_tensor = transform_2(augmented_image)
                    images.append(aug_img_tensor)
                    labels.append(label)
                    label_counts[cancer_annot_str[label]] += 1
            else:
                for _ in range(1):
                    augmented_image = transform_1(augmented_image)
                    aug_img_tensor = transform_2(augmented_image)
                    images.append(aug_img_tensor)
                    labels.append(label)
                    label_counts[cancer_annot_str[label]] += 1

    return torch.stack(images), torch.tensor(labels)
def label_names(filename):
    annotation_dir = "C:/Users/anant/Downloads/final_year_pro_datasets/PKG - CDD-CESM/CDD-CESM/Radiology-manual-annotations.csv"
    df = pd.read_csv(annotation_dir)
    filename = filename.split(".")[0]
    cancer_annot = {"Normal": 0, "Benign": 0, "Malignant":1}
    row = df[df["Image_name"] == filename.strip()]
    labels = row.values[-1][-1]
    labels = cancer_annot[labels]
    return labels, row.values[-1][-1]
def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cdd':
        data_dir = "C:/Users/anant/Downloads/final_year_pro_datasets/PKG - CDD-CESM/CDD-CESM/higher_energy/L_MLO"
        images, labels = preprocess_images(data_dir)
        dataset = CustomImageDataset(images, labels)
        # Define the split ratio
        train_ratio = 0.8
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        user_groups = cnn_iid(train_dataset, args.num_users)
        train_dataset, test_dataset, user_groups = dataset,test_dataset,user_groups
    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def fednova_aggregation(global_model, local_weights, num_users, local_data_sizes):
    """
    Aggregates the local model updates using FedNova method.
    
    Parameters:
    - global_model: The global model
    - local_weights: List of local model weights
    - num_users: Number of users
    - local_data_sizes: List of data sizes for each user

    Returns:
    - Updated global model weights
    """
    # Initialize the global model weights
    global_weights = global_model.state_dict()

    # Compute total data size
    total_data_size = sum(local_data_sizes)

    # Initialize the numerator and denominator for FedNova
    fednova_numerator = {key: torch.zeros_like(value) for key, value in global_weights.items()}
    fednova_denominator = {key: torch.zeros_like(value) for key, value in global_weights.items()}

    for i in range(num_users):
        # Load local model weights
        local_weight = local_weights[i]

        # Get local data size
        local_data_size = local_data_sizes[i]

        # Normalize local weights
        normalized_weights = {key: (local_weight[key] - global_weights[key]) / local_data_size for key in local_weight.keys()}

        # Accumulate the normalized weights
        for key in fednova_numerator.keys():
            fednova_numerator[key] += normalized_weights[key] * local_data_size
            fednova_denominator[key] += local_data_size

    # Update global weights using the normalized updates
    for key in global_weights.keys():
        global_weights[key] += fednova_numerator[key] / fednova_denominator[key]

    return global_weights
