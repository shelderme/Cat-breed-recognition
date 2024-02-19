import shutil
import os
import random
import cv2
import numpy as np
from config import*

def preprocess_image(image_path, target_width=224, target_height=224):

    image = cv2.imread(image_path) 

    # Если не RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Нормализация значений пикселей
    normalized_image = image / 255.0

    # Изменение размера изображения
    resized_image = cv2.resize(normalized_image, (target_width, target_height))
    
    # Приведение к формату NumPy array
    image_array = np.array(resized_image)
    
    return image_array


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
            train_labels.append(class_mapping[file_name])
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





# def copy_raw_data():
#     percentage_for_train = 70
#     add_percentage = 50
#     train_data_processed = []
#     validation_data_processed = []
#     test_data_processed = []
#     # Получить список файлов в исходной папке
#     files = os.listdir(source_folder)

#     for file_name in files:
#         # Перемещаем 70(1000) процентов файлов из общего датасета для одной породы в папку для train
#         data = os.listdir(source_folder + '/' + file_name)
#         train_files_num = 700 #int(len(data) * (percentage_for_train / 100))
#         files_to_move = random.sample(data, train_files_num)

#         i = 0
#         for jpgs in files_to_move:
#             source_path = os.path.join(source_folder + '/' + file_name, jpgs)
#             destination_path = os.path.join(destination_folder + '/train/' + file_name, jpgs)
#             shutil.move(source_path, destination_path)
#             # Меняем имя файла
#             os.rename(destination_path, destination_folder + '/train/' + file_name + f'/{i}.jpg')
#             # Обрабатываем изображение
#             train_data_processed.append(preprocess_image(destination_folder + '/train/' + file_name + f'/{i}.jpg'))
#             i += 1

#         i = 0
#         # Перемещаем половину оставшихся(test data)
#         data = os.listdir(source_folder + '/' + file_name)
#         test_files_num = 150 # int(len(data) * (add_percentage / 100))
#         files_to_move = random.sample(data, test_files_num)

#         for jpgs in files_to_move:
#             source_path = os.path.join(source_folder + '/' + file_name, jpgs)
#             destination_path = os.path.join(destination_folder + '/test/' + file_name, jpgs)
#             shutil.move(source_path, destination_path)

#             os.rename(destination_path, destination_folder + '/test/' + file_name + f'/{i}.jpg')
#             # Обработка
#             test_data_processed.append(preprocess_image(destination_folder + '/test/' + file_name + f'/{i}.jpg'))
#             i += 1

        

#         # Перемещаем остаток(validation data) - то же самое количество, что и в train data
#         data = os.listdir(source_folder + '/' + file_name)
#         files_to_move = random.sample(data, test_files_num)

#         i = 0
#         for jpgs in files_to_move:
#             source_path = os.path.join(source_folder + '/' + file_name, jpgs)
#             destination_path = os.path.join(destination_folder + '/validation/' + file_name, jpgs)
#             shutil.move(source_path, destination_path)
#             os.rename(destination_path, destination_folder + '/validation/' + file_name + f'/{i}.jpg')
#             # Обработка
#             validation_data_processed.append(preprocess_image(destination_folder + '/validation/' + file_name + f'/{i}.jpg'))
#             i += 1
        
#     # Сохраняем данные в файл
#     np.save('train_data.npy', np.array(train_data_processed))
#     np.save('test_data.npy', np.array(test_data_processed))
#     np.save('validation_data.npy', np.array(validation_data_processed))


# def is_folder_empty(folder_path):
#     # Получить список файлов в папке
#     files = os.listdir(folder_path)

#     # Проверить, пуста ли папка
#     if len(files) == 0:
#         return True
#     else:
#         return False
    


# def clear_all_data():
#     files = os.listdir(destination_folder)
#     for file_name in files:
#         data = os.listdir(destination_folder + '/' + file_name)
#         for files_ in data:
#             data_ = os.listdir(destination_folder + '/' + file_name + '/' + files_)
#             for jpegs in data_:
#                 os.remove(destination_folder + '/' + file_name + '/' + files_ + '/' + jpegs)


# def labeling(data):
    
#     labeled_data = []

#     breeds = class_mapping
#     for breed_name in breeds:
#         # Отмечаем котов для train_data
#         cats_count = len(os.listdir(destination_folder + data + breed_name))
#         cats_list = [class_mapping[breed_name]] * cats_count
#         labeled_data += cats_list

#     return labeled_data



