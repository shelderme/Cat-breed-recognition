import matplotlib.pyplot as plt
import pickle


# Загрузка истории обучения из файла
with open('training_history.pkl', 'rb') as file:
    history = pickle.load(file)

# График точности
plt.plot(history.history['accuracy'], label='Точность (Обучение)')
plt.plot(history.history['val_accuracy'], label='Точность (Валидация)')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()

# График потерь
plt.plot(history.history['loss'], label='Потери (Обучение)')
plt.plot(history.history['val_loss'], label='Потери (Валидация)')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.show()



