from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D,MaxPooling2D
from keras.optimizers import Adam, Nadam
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


from config import*
from preprocessing import*
import pickle

class_names = {'bengal', 'maine_coon', 'ragdoll', 'siamese', 'tortoiseshell'}

def create_model():
    # Загрузка данных
    train_data = np.load('train_data.npy')
    train_labels = np.load('train_labels.npy')
    # train_data - [array, ...]
    #[bengal, bengal...]

    validation_data = np.load('valid_data.npy')
    validation_labels = np.load('valid_labels.npy')

    test_data = np.load('test_data.npy')
    test_labels = np.load('test_labels.npy')

    # Загрузка предварительно обученной модели VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Замораживаем веса предварительно обученной части модели
    base_model.trainable = False

    # Создаем модель
    custom_model = Sequential([
        base_model,
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')  # 5 - количество классов (пород кошек)
    ])
  
    
##############################################################
    # Компилируем модель
    optimizer=Adam()
    #optimizer = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    
    custom_model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train_labels = labeling('/train/')
    # test_labels = labeling('/test/')
    # validation_labels = labeling('/validation/')

    # # Преобразование в NDArray
    # train_data_ND = np.array(train_data)
    # validation_data_ND = np.array(validation_data)
    # test_data_ND = np.array(test_data)
    # train_labels_ND = np.array(train_labels)
    # validation_labels_ND = np.array(validation_labels)
    # test_labels_ND = np.array(test_labels)


    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    # Обучаем модель, а затем сохраняем историю обучения в переменную history, затем в файл
    history = custom_model.fit(train_data, train_labels, epochs=10, validation_data=(validation_data, validation_labels), callbacks=[early_stopping])
    with open('training_history.pkl', 'wb') as file:
        pickle.dump(history, file)


    test_loss, test_accuracy = custom_model.evaluate(test_data, test_labels)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print(f'Test Loss: {test_loss * 100:.2f}%')

    # Сохраняем модель
    custom_model.save('cat_breed_model.h5')

    