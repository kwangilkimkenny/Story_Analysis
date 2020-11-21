# LSI(=LSA) 토픽모델링 기술 : 
#   Latent Dirichlet Allocation
#   토픽 모델링 기법 중 하나로 문헌-용어 행렬에서 문헌별 주제분포와 주제별 단어분포를 찾아주는 기술

import time
import pickle
from gensim import corpora, models

# 시간 출력 함수
def show_par(input):
    time = str(input.tm_min) + 'min ' + str(input.tm_sec) + 'sec'
    return time 

print("start!")
start_time = time.time() # 시작시간 저장


# 1단계 : 데이터 전처리 과정
with open ('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/fin_texts.pickle','rb') as fintext:
    final_texts = pickle.load(fintext)
# print(final_texts)

check_time = time.time()
print("데이터 전처리된 리스트 불러오기 WorkingTime: {}".format(show_par(time.gmtime(check_time - start_time))))
print("----------------------------------------------------------------------------------------")


# 2단계 : 사전 만들기
dictionary = corpora.Dictionary(final_texts)#make dictionary  문장을 수치화 하여 사전으로 만듬
dictionary.save('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/similerlist.dict')  
# save as binary file at the dictionary at local directory 일단 사전을 만든다. 
dictionary.save_as_text('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/similerlist_text.dict')  
# save as text file at the local directory

check_time = time.time()
print("딕션너리 저장 WorkingTime: {}".format(show_par(time.gmtime(check_time - start_time))))
print("----------------------------------------------------------------------------------------")


# 3단계 : 벡터화하기 - 만들어진 사전 정보를 통해서 벡터화 하기
corpus = [dictionary.doc2bow(text) for text in final_texts] 
#문장을 수치화하여 코퍼스 리스트로 만듬
# doc2bow 문서데이터를 수치화
corpora.MmCorpus.serialize('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/similerlist.mm', corpus) 
# save corpus at local directory Matrix Market
# format으로 저장

# tfidf로 벡터를 변환하기 
tfidf = models.TfidfModel(corpus) 
# initialize a model 초기화
corpus_tfidf = tfidf[corpus]  
# 코퍼스 모델을 tfidf공간에 맵핑

# 4단계 : LSI 모델링
Define_Num_Topic = 80
#토픽수 정의
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=Define_Num_Topic) 
# LSI 초기화
# corpus_lsi = lsi[corpus_tfidf] 
# # create a double wrapper over the original corpus

#모델 저장 및 로드
lsi.save('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/model.lsi')  

check_time = time.time()
print("LSI 모델링 WorkingTime: {}".format(show_par(time.gmtime(check_time - start_time))))
print("----------------------------------------------------------------------------------------")