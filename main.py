import pandas as pd
import pymorphy2
import nltk
from nltk import ngrams
from nltk.corpus import stopwords

import joblib

nltk.download('stopwords')
stop = stopwords.words('russian')
# Дропну "не" из списка, так как сильно влияет на смысл
stop.pop(stop.index('не'))

key_words = {'Консультация КЦ': ['оператор', 'лин', 'горяч', 'чат', 'информац', 'ден'],
 'Компетентность продавцов/ консультантов': ['услуг',
  'знат',
  'отказа',
  'дополнительн',
  'витрин',
  'коробк',
  'менеджер',
  'итог',
  'сдела',
  'сам'],
 'Электронная очередь': ['талон',
  'электрон',
  'табл',
  'номер',
  'термина',
  'взят',
  'приложен',
  'итог',
  'брат',
  'должн'],
 'Доступность персонала в магазине': ['отсутствова', 'мест', 'найт', 'долг'],
 'Вежливость сотрудников магазина': ['охранник',
  'груб',
  'хамств',
  'хам',
  'общен',
  'хамск',
  'менеджер',
  'попрос',
  'виде',
  'сам'],
 'Обслуживание на кассе': ['кассир', 'хамск'],
 'Обслуживание продавцами/ консультантами': ['вниман',
  'дел',
  'занят',
  'обраща',
  'ход',
  'сидет'],
 'Время ожидания у кассы': ['больш', 'огромн', 'кассир', 'долг', 'онлайн']}

def text_clean(text:pd.Series, method='L', rm_stop=False, rm_eng=False, rm_numb=False): #method='L', rm_stop=True, rm_eng=True, rm_numb=True
  # Понижение регистра
  text = text.str.lower()

  # Убираю пунктуацию, учитывая подобные ситуации: "привет,Алексей"
  text = text.str.replace(r'[^\w\s]', ' ', regex=True)
  text = text.str.replace(r'\s{2,}', ' ', regex=True)

  # Убираю цифры из текста
  if rm_numb:
    text = text.str.replace(r'[0-9]+', '', regex=True)
  # Убираю английские символы
  if rm_eng:
    text = text.str.replace(r'[A-Za-z]+', '', regex=True)
  # Убираю стоп-слова
  if rm_stop:
    text = text.apply(lambda phrase: " ".join([word for word in phrase.split() if word not in stop]))
  if method == 'L':
    return lemmatisation(text)
  return text

def lemmatisation(data):
  morph = pymorphy2.MorphAnalyzer(lang='ru')
  X_train_lemm = data.apply(lambda x: ' '.join([morph.parse(y)[0].normal_form for y in x.split()]))
  return X_train_lemm

def keywords_data_column(keys_dict: dict, X: pd.Series, dataset: pd.DataFrame):
  for cls, words in keys_dict.items():
    if not words:
      continue
    filter_conc = X.str.contains(words[0])
    for word in words:
      filter_conc = filter_conc | X.str.contains(word)
    dataset[cls + '_meta'] = 0
    dataset.loc[filter_conc, cls + '_meta'] = 1
  return dataset

def add_agression(X: pd.Series, dataset: pd.DataFrame):
  dataset['agression'] = 0
  filter = X.str.contains('язв')
  patterns = ['язв', 'сук', 'хрен', 'хер', 'хуел', 'хом', 'хам', 'цирк',  'еба', 'пизд', 'пиzд', 'долбо','[!?]{2,}']
  for pat in patterns:
    filter = filter | X.str.contains(pat)
  dataset.loc[filter, 'agression'] = 1
  return dataset

def do_n_grams(string, n=6):
  ngram = ngrams(string, n)
  new_list = []
  for grams in ngram:
    new_list.append(''.join(grams))
  return ' '.join(new_list)

def process_text(text):
  # Предобработка. Понижаю регистр, убираю пунктуацию, стоп-слова, извлекаю леммы.
  text = text_clean(text=text, method='L')

  # Создаем датафрейм с фичами
  # new_data = pd.DataFrame(text.index, columns=['drop_me']).set_index('drop_me')
  # new_data = keywords_data_column(keys_dict=key_words, X=text, dataset=new_data)
  # new_data = add_agression(X=text, dataset=new_data)

  #хстак с векторизованным признаковым пространством
  text = text.apply(lambda x: do_n_grams(x, n=3))
  feature_tfidf_vectorizer  = joblib.load('vectorizer.joblib')
  X_tfidf = feature_tfidf_vectorizer.transform(text)
  # x_text_tfidf = hstack((X_tfidf, new_data.to_numpy()))
  return X_tfidf


