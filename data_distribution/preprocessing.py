import shutil
import os
import random
import cv2
import numpy as np
from config import*
import imgaug.augmenters as iaa
import sys

def preprocess_image(image_path, target_width=224, target_height=224):

    image = cv2.imread(image_path) 

    # Если не RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Нормализация значений пикселей
    resized_image = cv2.resize(image, (target_width, target_height))

    # Приведение к формату NumPy array и типу uint8
    image_array = np.array(resized_image, dtype=np.uint8)
    return image_array


def augment_image(image_path, target_width = 224, target_height = 224):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Отражение по горизонтали с sвероятностью 50%
        iaa.Affine(rotate=(-20, 20)),  # Вращение на случайный угол от -20 до 20 градусов
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Добавление случайного размытия
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # Добавление случайного шума
    ], random_order=True)  # Применять аугментации в случайном порядке
    new_image = seq(image=image)
    resized_image = cv2.resize(new_image, (target_width, target_height))
    return resized_image


def data_collection():
    breeds = os.listdir(dataset_path)
    print(f'Dataset content:')

    for file_name in breeds:   
        print(file_name, '\t(' + str(len(os.listdir(dataset_path + '/' + file_name))) + ' files)')

    print("\nSelect the number of photos on which the model will be trained:", end=' ')
    photos_number = int(input())
    print('*This number will apply to one breed, that is, if you enter 500, then 500 photos will be selected for each breed.'
f'\n(Total number of photos: {photos_number} * {len(breeds)} = {photos_number * len(breeds)})')
    
    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    valid_data = []
    valid_labels = []

    for file_name in breeds:
        favorite_files = random.sample(os.listdir(dataset_path + '/' + file_name), photos_number)
        train_files = random.sample(favorite_files, int(photos_number * 0.7))
        for files in train_files:
            preprocessed_image = preprocess_image(dataset_path + '\\' + file_name + '\\' + files)
            train_data.append(preprocessed_image)
            train_labels.append([class_mapping[file_name]])
            # augmented_image = augment_image(dataset_path + '\\' + file_name + '\\' + files)
            # train_data.append(augmented_image)
            # train_labels.append([class_mapping[file_name]])
            
            favorite_files.remove(files)
        
        test_files = random.sample(favorite_files, int(photos_number * 0.15))
        for files in test_files:
            preprocessed_image = preprocess_image(dataset_path + '\\' + file_name + '\\' + files)
            test_data.append(preprocessed_image)
            test_labels.append(class_mapping[file_name])
            favorite_files.remove(files)

        for files in favorite_files:
            preprocessed_image = preprocess_image(dataset_path + '\\' + file_name + '\\' + files)
            valid_data.append(preprocessed_image)
            valid_labels.append(class_mapping[file_name])
    
    np.save(f'train_data.npy', np.array(train_data))
    np.save(f'train_labels.npy', np.array(train_labels))

    np.save(f'test_data.npy', np.array(test_data))
    np.save(f'test_labels.npy', np.array(test_labels))

    np.save(f'valid_data.npy', np.array(valid_data))
    np.save(f'valid_labels.npy', np.array(valid_labels))





