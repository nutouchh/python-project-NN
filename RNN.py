import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences

# Задаем параметры обработки текста
max_words = 10000  # максимальное количество слов в словаре
max_len = 100  # максимальное количество слов в предложении
embedding_size = 128  # размерность векторного пространства

# Считываем текстовые данные
texts1 = np.array(['это первый текст', 'это второй текст', 'это третий текст'])
texts2 = np.array(['это первый текст', 'это новый второй текст', 'это новый третий текст'])
labels = np.array([1, 0, 0])  # 1 - плагиат, 0 - нет

text = np.concatenate((texts1, texts2), axis=0)

# Создаем токенизатор и обучаем его на текстовых данных
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text)

# Преобразуем тексты в числовые последовательности
sequences1 = tokenizer.texts_to_sequences(texts1)
sequences2 = tokenizer.texts_to_sequences(texts2)

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

print(X2)
print(labels)
# Обучаем модель на наших данных
model.fit(np.array([X1, X2]), labels, epochs=10, batch_size=128)

# Определяем процент плагиата
y_pred = model.predict(np.array([X1, X2]))
percent_plagiarism = sum(y_pred > 0.5) / len(y_pred)
print(f'Процент плагиата: {percent_plagiarism}')
