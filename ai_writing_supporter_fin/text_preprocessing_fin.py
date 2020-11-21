import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import multiprocessing
import itertools
import parmap
import pickle
import time
import re

# 시간 출력 함수
def show_par(input):
    time = str(input.tm_min) + 'min ' + str(input.tm_sec) + 'sec'
    return time 

print("start!")
start_time = time.time() # 시작시간 저장

num_cores = multiprocessing.cpu_count() # 6

#데이터 전처리 
def cleaning(datas):

    fin_datas = []

    for data in datas:
        # 영문자 이외 문자는 공백으로 변환
        only_english = re.sub('[^a-zA-Z]', ' ', data)
    
        # 소문자 변환
        no_capitals = only_english.lower().split()
    
        # 불용어 제거
        stops = set(stopwords.words('english'))
        no_stops = [word for word in no_capitals if not word in stops]
    
        # 어간 추출
        stemmer = nltk.stem.SnowballStemmer('english')
        stemmer_words = [stemmer.stem(word) for word in no_stops]
        # see, saw, seen/run, running, ran
        # 위 예시처럼 어형이 과거형이든 미래형이든 하나의 단어로 취급하기 위한 처리작업
    
        # 공백으로 구분된 문자열로 결합하여 리스트 결과 반환
        fin_words = (' '.join(stemmer_words)).split()

        # 데이터를 리스트에 추가 
        fin_datas.append(fin_words)

    return fin_datas

#remove words those appear only once 단 한번만 나타나는 word 삭제
def delete_one_word(texts_list, one_word_list):

    for word in one_word_list:

        flag = False
        for texts in texts_list:
            for text in texts:
                if text == word:
                    # print(text)
                    texts.remove(text)
                    flag = True
                    break

            if flag == True:
                break

    return texts_list    

with open('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/essay_source.pickle', 'rb') as f:
        essay_source_data = pickle.load(f)
documents = essay_source_data

check_time = time.time()
print("에세이 불러오기 WorkingTime: {}".format(show_par(time.gmtime(check_time - start_time))))
print("----------------------------------------------------------------------------------------")


splited_documents = np.array_split(documents, num_cores) 
splited_documents = [documents.tolist() for documents in splited_documents]
splited_documents = parmap.map(cleaning, splited_documents, pm_pbar=True, pm_processes=num_cores)
texts = []
for documents in splited_documents:
    texts.extend(documents)

check_time = time.time()
print("cleaning 데이터 WorkingTime: {}".format(show_par(time.gmtime(check_time - start_time))))
print("----------------------------------------------------------------------------------------")


all_tokens = list(itertools.chain.from_iterable(texts))
# print(all_tokens[:100])
list_texts = np.array(texts).flatten().tolist()
# print(final_texts[:100])
# https://winterj.me/list_of_lists_to_flatten/

frequency = {}
for word in all_tokens:
    frequency[word] = frequency.get(word,0) + 1
    
frequency_list = frequency.keys()
one_word = []

for word in frequency_list:
    if frequency[word] == 1:
        one_word.append(word)

check_time = time.time()
print("one_word 리스트 생성 WorkingTime: {}".format(show_par(time.gmtime(check_time - start_time))))
print("----------------------------------------------------------------------------------------")


splited_list_texts = np.array_split(list_texts, num_cores) 
splited_list_texts = [list_texts.tolist() for list_texts in splited_list_texts]
splited_list_texts = parmap.map(delete_one_word, splited_list_texts, one_word, pm_pbar=True, pm_processes=num_cores)
final_texts = []
for list_texts in splited_list_texts:
    final_texts.extend(list_texts)
# print(final_texts[:100])

check_time = time.time()
print("one_word 제거된 리스트 WorkingTime: {}".format(show_par(time.gmtime(check_time - start_time))))
print("----------------------------------------------------------------------------------------")


with open ('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/fin_texts.pickle','wb') as fintext:
    pickle.dump(final_texts, fintext)

check_time = time.time()
print("리스트 저장 WorkingTime: {}".format(show_par(time.gmtime(check_time - start_time))))
print("----------------------------------------------------------------------------------------")
