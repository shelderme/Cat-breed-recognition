from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D,MaxPooling2D
from keras.optimizers import Adam, Nadam
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import regularizers
import seaborn as sns
import dill

from config import*
from preprocessing import*
import pickle

class_names = {'bengal', 'maine_coon', 'ragdoll', 'siamese', 'tortoiseshell'}

def create_model():
    # Загрузка данных
    train_data = np.load('train_data.npy')
    train_labels = np.load('train_labels.npy')
    validation_data = np.load('valid_data.npy')
    validation_labels = np.load('valid_labels.npy')
    test_data = np.load('test_data.npy')
    test_labels = np.load('test_labels.npy')

    # Генераторы данных
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow(
        train_data, train_labels,
        batch_size=32,
        shuffle=True
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow(
        validation_data, validation_labels,
        batch_size=32,
        shuffle=False
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow(
        test_data, test_labels,
        batch_size=32,
        shuffle=False
    )

    # Загрузка предварительно обученной модели EfficientNetB0
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True
    fine_tune_at = 15
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Создание модели
    custom_model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(5, activation='softmax')
    ])

    # Компиляция модели
    optimizer = Adam(learning_rate=0.0001)
    custom_model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Обучение модели
    history = custom_model.fit(
        train_generator,
        steps_per_epoch=len(train_data) // 32,  # количество шагов для обучения
        epochs=10,
        validation_data=validation_generator,
        validation_steps=len(validation_data) // 32,  # количество шагов для валидации
        callbacks=[early_stopping]
    )

    # Сохранение истории обучения
    history_dict = history.history
    with open('history.pkl', 'wb') as file:
        pickle.dump(history_dict, file)

    # Оценка модели на тестовом наборе
    test_loss, test_accuracy = custom_model.evaluate(test_generator, steps=len(test_data) // 32)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print(f'Test Loss: {test_loss * 100:.2f}%')

    # Сохранение модели
    custom_model.save('cat_breed_model.keras')

create_model()
