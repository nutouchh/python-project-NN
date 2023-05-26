from pymystem3 import Mystem
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')

m = Mystem()

with open("datasets/paraphrase/paraphrase1.txt", encoding='utf-8') as file:
    text = file.read()

new_text = " ".join(m.lemmatize(text))


corpus = re.sub(r'[^а-яА-ЯёЁ ]', ' ', new_text)
corpus = " ".join(corpus.split())
# print(corpus)

corpus = corpus.split()
stopword = set(stopwords.words('russian'))
# print(stop_words)
count_vect = CountVectorizer(stop_words=list(stopword)) # stopwords - список стоп-слов
# bow, от англ. bag of words
bow = count_vect.fit_transform(corpus)
# словарь уникальных слов
words = count_vect.get_feature_names_out()
# print(words)

to_file = '\n'.join(words)
with open('datasets/paraphrase/paraphrase1done.txt', 'w') as f:
    f.write(to_file)