import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Задаем параметры обработки текста
max_words = 10000  # максимальное количество слов в словаре
max_len = 100  # максимальное количество слов в предложении
embedding_size = 128  # размерность векторного пространства

# Считываем текстовые данные
text1 = "Это первый текст для проверки плагиата."
text2 = "Это второй текст для проверки плагиата, но он похож на первый."

# Создаем токенизатор и обучаем его на текстовых данных
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts([text1, text2])

# Преобразуем тексты в числовые последовательности
sequences1 = tokenizer.texts_to_sequences([text1])
sequences2 = tokenizer.texts_to_sequences([text2])

# Ограничиваем длину последовательностей и добавляем нули, если они короче
X1 = pad_sequences(sequences1, maxlen=max_len)
X2 = pad_sequences(sequences2, maxlen=max_len)

# Создаем матрицу эмбеддингов для входных данных
embedding_matrix = np.random.rand(max_words, embedding_size)

# Создаем модель RNN
model = Sequential()
model.add(Embedding(max_words, embedding_size, input_length=max_len))
model.add(LSTM(256, dropout=0.5))
model.add(Dense(1, activation='sigmoid'))

# Компилируем модель
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучаем модель на наших данных
model.fit(X1, np.array([0]), epochs=10, batch_size=128)
output1 = model.predict(X1)

model.fit(X2, np.array([0]), epochs=10, batch_size=128)
output2 = model.predict(X2)

# Определяем процент плагиата на основе сравнения выходных данных
plagiarism_percentage = abs(output1 - output2) * 100

print("Процент плагиата: %.2f%%" % plagiarism_percentage)
