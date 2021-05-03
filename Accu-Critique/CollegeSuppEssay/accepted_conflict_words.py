
#conflict
import pickle
import nltk

# 다운로드 이미 완료, 실행시 사용하지 않음
# punkt = nltk.download('punkt')
# with open('nltk_punkt.pickle', 'wb') as f:
#     pickle.dump(punkt, f, pickle.HIGHEST_PROTOCOL)

with open('./data_accepted_st/nltk_punkt.pickle', 'rb') as f:
    punkt = pickle.load(f)

# 다운로드 이미 완료, 실행시 사용하지 않음
# vader_lexicon = nltk.download('vader_lexicon')
# with open('vader_lexicon.pickle', 'wb') as f:
#     pickle.dump(punkt, f, pickle.HIGHEST_PROTOCOL)

with open('./data_accepted_st/vader_lexicon.pickle', 'rb') as f:
    vader_lexicon = pickle.load(f)

# 다운로드 이미 완료, 실행시 사용하지 않음
# averaged_perceptron_tagger  = nltk.download('averaged_perceptron_tagger')
# with open('averaged_perceptron_tagger.pickle', 'wb') as f:
#     pickle.dump(punkt, f, pickle.HIGHEST_PROTOCOL)
    
with open('./data_accepted_st/averaged_perceptron_tagger.pickle', 'rb') as f:
    averaged_perceptron_tagger = pickle.load(f)

from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import pandas as pd
from pandas import DataFrame as df
from mpld3 import plugins, fig_to_html, save_html, fig_to_dict
from tqdm import tqdm
import numpy as np
import json
from tensorflow.keras.preprocessing.text import text_to_word_sequence

#character, setting
import numpy as np
import gensim
import nltk

#다운로드 이미 완료, 실행시 사용하지 않음
# stopwords = nltk.download('stopwords')
# with open('stopwords.pickle', 'wb') as f:
#     pickle.dump(stopwords, f, pickle.HIGHEST_PROTOCOL)
    
with open('./data_accepted_st/stopwords.pickle', 'rb') as f:
    stopwords = pickle.load(f)

from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from nltk.tokenize import sent_tokenize
import multiprocessing
import os
from pathlib import Path
import io
from gensim.models import Phrases
from textblob import TextBlob
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline

# 다운로드했기 때문에 실행시는 사용하지 않음
# tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
# model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

# # save tokenizer ---> 계산속도 줄이기위해서 미리 저장, 저장했기 때문에 실행시는 사용하지 않음
# with open('data_tokenizer.pickle', 'wb') as f:
#     pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)

# # save model ---> 계산속도 줄이기위해서 미리 저장, 저장했기 때문에 실행시는 사용하지 않음
# with open('data_model.pickle', 'wb') as g:
#     pickle.dump(model, g, pickle.HIGHEST_PROTOCOL)

# open tokenizer
with open('./data_accepted_st/data_tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# open model  --------> 이거새으 400MB 가 넘어서 git에 올라가지 않음, 그럴경우 아래 코드의 주석을 풀어서 사용해야 함
######----- model 주석 해제하여 사용할 것  ----####
# model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
#############################################

with open('./data_accepted_st/data_model.pickle', 'rb') as g:
    model = pickle.load(g)


goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
)


#데이터 전처리 
def cleaning(datas):

    fin_datas = []

    for data in datas:
        # 영문자 이외 문자는 공백으로 변환
        only_english = re.sub('[^a-zA-Z]', ' ', data)
    
        # 데이터를 리스트에 추가 
        fin_datas.append(only_english)

    return fin_datas


def ai_plot_conf(essay_input_):
    #1.input essay
    input_text = essay_input_

    #########################################################################

    #2.유사단어를 추출하여 리스트로 반환
    def conflict_sim_words(text):

        essay_input_corpus = str(text) #문장입력
        essay_input_corpus = essay_input_corpus.lower()#소문자 변환

        sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화
        total_sentences = len(sentences)#토큰으로 처리된 총 문장 수
        total_words = len(word_tokenize(essay_input_corpus))# 총 단어수
        
        split_sentences = []
        for sentence in sentences:
            processed = re.sub("[^a-zA-Z]"," ", sentence)
            words = processed.split()
            split_sentences.append(words)

        skip_gram = 1
        workers = multiprocessing.cpu_count()
        bigram_transformer = Phrases(split_sentences)

        model = gensim.models.word2vec.Word2Vec(bigram_transformer[split_sentences], workers=workers, sg=skip_gram, min_count=1)

        model.train(split_sentences, total_examples=sum([len(sentence) for sentence in sentences]), epochs=100)
        
        #모델 설계 완료

        #표현하는 단어들을 리스트에 넣어서 필터로 만들고
        confict_words_list = ['clash', 'incompatible', 'inconsistent', 'incongruous', 'opposition', 'variance','vary', 'odds', 
                                'differ', 'diverge', 'disagree', 'contrast', 'collide', 'contradictory', 'incompatible', 'conflict',
                                'inconsistent','irreconcilable','incongruous','contrary','opposite','opposing','opposed',
                                'antithetical','clashing','discordant','differing','different','divergent','discrepant',
                                'varying','disagreeing','contrasting','at odds','in opposition','at variance' ]
        
        ####문장에 list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
        
        #우선 토큰화한다.
        retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
        token_input_text = retokenize.tokenize(essay_input_corpus)
        #print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
        #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
        filtered_chr_text = []
        for k in token_input_text:
            for j in confict_words_list:
                if k == j:
                    filtered_chr_text.append(j)
        
        #print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
        
        filtered_chr_text_ = set(filtered_chr_text) #중복제거
        filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
        #print (filtered_chr_text__) # 중복값 제거 확인
        
#         for i in filtered_chr_text__:
#             ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
#         char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 표현 수
#         char_count_ = len(filtered_chr_text__) #중복제거된  표현 총 수
            
#         result_char_ratio = round(char_total_count/total_words * 100, 2)

#         import pandas as pd

#         df_conf_words = pd.DataFrame(ext_sim_words_key, columns=['words','values']) #데이터프레임으로 변환
#         df_r = df_conf_words['words'] #words 컬럼 값 추출
#         ext_sim_words_key = df_r.values.tolist() # 유사단어 추출
        ext_sim_words_key = filtered_chr_text__

        #return result_char_ratio, total_sentences, total_words, char_total_count, char_count_, ext_sim_words_key
        return ext_sim_words_key



    #########################################################################
    # 3.유사단어를 문장에서 추출하여 반환한다.
    conflict_sim_words_ratio_result = conflict_sim_words(input_text)


    #########################################################################
    # 4.CONFLICT GRAPH EXPRESSION Analysis  -- 그래프로 그리기
    # conflict(input_text):
    contents = str(input_text)
    token_list_str = text_to_word_sequence(contents) #tokenize
    # 원본문장 단어 중복제거
    token_list_str_set = set(token_list_str)
    #print('token_list_str:', token_list_str)
    confict_words_list_basic = ['clash', 'incompatible', 'inconsistent', 'incongruous', 'opposition', 'variance','vary', 'odds', 
                            'differ', 'diverge', 'disagree', 'contrast', 'collide', 'contradictory', 'incompatible', 'conflict',
                            'inconsistent','irreconcilable','incongruous','contrary','opposite','opposing','opposed', 'fight',
                            'antithetical','clashing','discordant','differing','different','divergent','discrepant', 'beef', 'bone to pick',
                            'varying','disagreeing','contrasting','at odds','in opposition','at variance', 'different', 'bone of contention',
                            'battle', 'competition', 'combat', 'rivalry', 'strife', 'struggle', 'war', 'collision',
                            'contention', 'contest', 'emulation', 'encounter', 'engagement', 'fracas', 'fray', 'set-to',
                            'striving', 'tug-of-war', 'conflicted', 'conflicting', 'conflicts', 'disagreement','contrariety',
                            'friction', 'enmity', 'dissension', 'incongruity', 'rancor', 'resistance', 'hostility', 'hatred',
                            'discord', 'debate', 'controversy', 'dispute', 'agitation','matter','matter at hand','problem',
                            'point in question', 'question', 'dispute', 'issue', 'sore point', 'tender spot', 'quarrel', 'discord']

    confict_words_list = confict_words_list_basic + conflict_sim_words_ratio_result #유사단어를 계산결과 반영!
    #중복제거
    confict_words_list_set = set(confict_words_list)
    #print('confict_words_list:', confict_words_list_set)
    
    # 문장에 들어있는 추출된 conflict 단어들 : count_conflict_list ==================> conflict 단어가 없음(겹치는 단어 없나?)
    count_conflict_list = []
    for ittm in confict_words_list_set:
        if ittm in token_list_str_set:
            count_conflict_list.append(ittm)
            
    #print('문장에 들어있는 추출된 conflict 단어들:', count_conflict_list)
    
    # 전체문장에 들어있는 conflict 단어 수
    nums_conflict_words =  len(count_conflict_list)
    #print('전체문장에 들어있는 conflict 단어 수:', nums_conflict_words)

    return nums_conflict_words



def get_conflict_words():

    ratio_score_cnt = []
    path = "./data_accepted_st/ps_essay_evaluated.csv"
    data = pd.read_csv(path)
    #Score를 인덱스로 변환하여 데이터 찾아보기
    data.set_index('Score', inplace=True)
    for i in tqdm(data.index):
        if i is not None and i >= 4:
            get_essay = data.loc[i, 'Essay']

            input_ps_essay = get_essay
            result = ai_plot_conf(str(input_ps_essay))
            ratio_score_cnt.append(result)


    #print('emotion_counter:', emotion_counter)
    e_re = [y for x in ratio_score_cnt for y in x]
    # 중복감성 추출
    total_count = {}
    for i in e_re:
        try: total_count[i] += 1
        except: total_count[i]=1

    accepted_mean = round(sum(total_count) / len(total_count), 1)

    return accepted_mean


    