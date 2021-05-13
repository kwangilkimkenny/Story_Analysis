# Prompt Type: Overcoming a Challenge or ethical dilemma
# ppt p.24

# 1) Prompt Oriented Sentiments (50%)
# Main #1 총점의 40% (앞부분 40%에서 count): anger, annoyance, confusion, embarrassment, disappointment, disapproval, fear, nervousness, sadness, remorse
# Main #2 총점의 60% (뒷부분 60%에서 count): admiration, approval, caring, joy, gratitude, love, optimism, relief, pride
# *상반되는 main sentiment들의 ratio가 공존해야 좋은거 같아요 (challenge → overcome 으로 가는거니까요)

# Prompt Oriented Sentiments  -- 글속에 감정이 얼마나 표현되어 있는지 분석 - origin (Bert pre trained model 활용)
import numpy as np
import spacy
from collections import Counter
import re
import nltk
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load("en_core_web_sm")


import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import pandas as pd
from pandas import DataFrame as df
from mpld3 import plugins, fig_to_html, save_html, fig_to_dict
from tqdm import tqdm
import numpy as np
import json
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from wordcloud import WordCloud 
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
#%matplotlib inline

import matplotlib
from IPython.display import set_matplotlib_formats
matplotlib.rc('font',family = 'Malgun Gothic')
set_matplotlib_formats('retina')
matplotlib.rc('axes',unicode_minus = False)

import gensim
from gensim.models import Phrases

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
from gensim import corpora, models, similarities

from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from pprint import pprint

from intellectualEngagement import intellectualEnguagement

tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
)

# 이 함수는 두 군데 사용중 1)collegeSuppy.py 2)prompt_oriented_sentments.py, 하지만 나중에 분리해야 함. pmp 별로 개별 계산해야 하기떼문
def select_prompt_type(prompt_type):
    
    if prompt_type == 'Why us':
        pmt_typ = [""" 'Why us' school & major interest (select major, by college & department) """]
        pmt_sentiment = ['Admiration', 'Excitement', 'Pride', 'Realization', 'Curiosity']
    elif prompt_type == 'Intellectual interest':
        pmt_typ = [""" Intellectual interest """]
        pmt_sentiment = ['Curiosity', 'Realization']
    elif prompt_type == 'Meaningful experience & lesson learned':
        pmt_typ = ["Meaningful experience & lesson learned"]
        pmt_sentiment = ['Realization', 'Approval', 'Gratitude', 'Admiration']
    elif prompt_type ==  'Achievement you are proud of':
        pmt_typ = ["Achievement you are proud of"]
        pmt_sentiment = ['Realization', 'Approval', 'Gratitude', 'Admiration', 'Pride', 'Desire', 'Optimism']
    elif prompt_type ==  'Social issues: contribution & solution':
        pmt_typ = ["Social issues: contribution & solution"]
        # 0~16 번째는 에세이 입력의 40% 분석 적용, 그 이후부분은 60% 적용  --> 즉 [:16], [17:] 이렇게 나눌 것
        pmt_sentiment = ['Anger', 'annoyance', 'Fear', 'Disapproval', 'disgust', 'Disappointment','grief', 'nervousness', 'sadness', 'surprise', 'remorse', 'curiosity', 'embarrassment', 'Realization','Approval', 'Gratitude', 'Admiration','Admiration','Approval', 'Caring', 'Joy', 'Gratitude', 'Optimism','relief', 'Realization']
    elif prompt_type ==  'Summer activity':
        pmt_typ = ["Summer activity"]
        pmt_sentiment = ['Pride','Realization','Curiosity','Excitement','Amusement','Caring']
    elif prompt_type ==  'Unique quality, passion, or talent':
        pmt_typ = ["Unique quality, passion, or talent"]
        pmt_sentiment = ['Pride','Excitement','Amusement','Approval','Admiration','Curiosity']
    elif prompt_type ==  'Extracurricular activity or work experience':
        pmt_typ = [""]
        pmt_sentiment = ['Pride','Realization','Curiosity','Joy','Excitement','Amusement','Caring','Optimism']
    elif prompt_type ==  'Your community: role and contribution in your community':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Pride','Gratitude','Love']
    elif prompt_type ==  'College community: intended role, involvement, and contribution in college community':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Excitement','Pride','Gratitude']
    elif prompt_type ==  'Overcoming a Challenge or ethical dilemma':
        pmt_typ = ["Overcoming a Challenge or ethical dilemma"]
        pmt_sentiment = ['Anger','Fear','Disapproval','Disappointment','Confusion','Annoyed','Realization', 'Approval','Gratitude','Admiration','Relief','Optimism']
    elif prompt_type ==  'Culture & diversity':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Realization','Love','Approval','Pride','Gratitude']
    elif prompt_type ==  'Collaboration & teamwork':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Optimism','Gratitude','Love']
    elif prompt_type ==  'Creativity/creative projects':
        pmt_typ = [""]
        pmt_sentiment = ['Excitement','Realization','Curiosity','Desire','Amusement','Surprise']
    elif prompt_type ==  'Leadership experience':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Optimism','Gratitude','Love','Fear','Confusion','Nervousness']
    elif prompt_type ==  'Values, perspectives, or beliefs':
        pmt_typ = [""]
        pmt_sentiment = ['Anger','Fear','Disapproval','Disappointment','Realization','Approval','Gratitude','Admiration']
    elif prompt_type ==  'Person who influenced you':
        pmt_typ = [""]
        pmt_sentiment = ['Realization', 'Approval', 'Gratitude','Admiration','Caring','Love','Curiosity', 'Pride', 'Joy']
    elif prompt_type ==  'Favorite book/movie/quote':
        pmt_typ = [""]
        pmt_sentiment = ['Excitement', 'Realization', 'Curiosity','Admiration','Amusement','Joy']
    elif prompt_type ==  'Write to future roommate':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Realization','Love','Excitement','Approval','Pride','Gratitude','Amusement','Curiosity','Joy']
    elif prompt_type ==  'Diversity & Inclusion Statement':
        pmt_typ = [""]
        pmt_sentiment = ['Anger','Fear','Disapproval','Disappointment','Confusion','Annoyed','Realization','Approval','Gratitude','Admiration','Relief','Optimism']
    elif prompt_type ==  'Future goals or reasons for learning':
        pmt_typ = [""]
        pmt_sentiment = ['Realization','Approval','Gratitude','Admiration','Pride','Desire','Optimism']
    elif prompt_type ==  'What you do for fun':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration', 'Excitement', 'Curiosity', 'Amusement', 'Pride','Joy']
    else:
        pass

    # pmt_typ : prompt type 
    # pmt_sentiment : prompy type에 해당하는 sentiment
    return pmt_typ, pmt_sentiment


#데이터 전처리 
def cleaning(data):
    fin_data = []
    for data_itm in data:
        # 영문자 이외 문자는 공백으로 변환
        only_english = re.sub('[^a-zA-Z]', ' ', data_itm)
        lists_re_ = re.sub(r"^\s+|\s+$", "", only_english) # 공백문자 제거
        only_english_ = lists_re_.rstrip('\n')
        # 데이터를 리스트에 추가 
        fin_data.append(only_english_)
    return fin_data


# txt 문서 정보 불러오기 : 대학정보
def open_data(select_college):
    # 폴더 구조, 대학이름 입력 명칭을 통일해야 함
    file_path = "./college_info/college_dataset/"
    college_name = select_college
    file_name = "_college_general_info.txt"
    # file = open("./college_info/colleges_dataset/brown_college_general_info.txt", 'r')
    file = open(file_path + college_name + file_name, 'r')
    lists = file.readlines()
    file.close()
    lists_re =  cleaning(lists) # 영어 단어가 아닌 것은 삭제
    result = ' '.join(lists_re) # 문장으로 합치기
    # 소문자 변환
    result_ = result.lower()
    #print("입력문장 불러오기 확인 : ", result_)
    return result_


# txt 문서 정보 불러오기 : 선택한 전공관련 정보 추출 
# 입력값은 대학, 전공 ex) 'Browon', 'AfricanStudies'
def open_major_data(select_college, select_major):
    # 폴더 구조, 대학이름 입력 명칭을 통일해야 함
    file_path = "./major_info/major_dataset/"
    college_name = select_college
    mjr_name = select_major
    file_name = "_major_info.txt"
    # file = open("./major_info/major_dataset/Brown_AfricanStudies_major_info.txt", 'r')
    file = open(file_path + college_name + "_" + mjr_name + file_name, 'r')
    lists = file.readlines()
    file.close()
    doc = ' '.join(lists)
    return lists


# 대학관련 정보의 토픽 키워드 추출하여 WordCloud로 구현
def general_keywords(College_text_data):
    tokenized = nltk.word_tokenize(str(College_text_data))
    #print('tokenized:', tokenized)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if(pos[:2] == 'NN')]
    count = Counter(nouns)
    words = dict(count.most_common())
    #print('words:', words)
    # 가장 많이 등장하는 단어를 추려보자. 
    wordcloud = WordCloud(background_color='white',colormap = "Accent_r",
                            width=1500, height=1000).generate_from_frequencies(words)

    plt.imshow(wordcloud)
    plt.axis('off')
    gk_re = plt.show()

    return gk_re


def Prompt_Oriented_Sentiments_analysis(essay_input):
    ########## 여기서는 최초 입력 에세이를 적용한다. input_text !!!!!!!!
    re_text = essay_input.split(".")

    #데이터 전처리 
    def cleaning(datas):

        fin_datas = []

        for data in datas:
            # 영문자 이외 문자는 공백으로 변환
            only_english = re.sub('[^a-zA-Z]', ' ', data)
        
            # 데이터를 리스트에 추가 
            fin_datas.append(only_english)

        return fin_datas

    texts = cleaning(re_text)

    #분석된 감정만 추출
    emo_re = goemotions(texts)

    emo_all = []
    for list_val in range(0, len(emo_re)):
        #print(emo_re[list_val]['labels'],emo_re[list_val]['scores'])
        #mo_all.append((emo_re[list_val]['labels'],emo_re[list_val]['scores'])) #KEY, VALUE만 추출하여 리스트로 저장
        #emo_all.append(emo_re[list_val]['scores'])
        emo_all.append((emo_re[list_val]['labels']))
        
    #추출결과 확인 
    # emo_all

    # ['sadness'],
    #  ['anger'],
    #  ['admiration', 'realization'],
    #  ['admiration', 'disappointment'],
    #  ['love'],
    #  ['sadness', 'neutral'],
    #  ['realization', 'neutral'],
    #  ['neutral'],
    #  ['optimism'],
    #  ['neutral'],
    #  ['excitement'],
    #  ['neutral'],
    #  ['neutral'],
    #  ['caring'],
    #  ['gratitude'],
    #  ['admiration', 'approval'], ...

    from pandas.core.common import flatten #이중리스틀 FLATTEN하게 변환
    flat_list = list(flatten(emo_all))

    # ['neutral',
    #  'neutral',
    #  'sadness',
    #  'anger',
    #  'admiration',
    #  'realization',
    #  'admiration',
    #  'disappointment',


    #중립적인 감정을 제외하고, 입력한 문장에서 다양한 감정을 모두 추출하고 어떤 감정이 있는지 계산해보자
    unique = []
    for r in flat_list:
        if r == 'neutral':
            pass
        else:
            unique.append(r)

    #중립감정 제거 및 유일한 감정값 확인
    #unique
    unique_re = set(unique) #중복제거

    ############################################################################
    # 글에 표현된 감정이 얼마나 다양한지 분석 결과!!!¶
    # print("====================================================================")
    # print("에세이에 표현된 다양한 감정 수:", len(unique_re))
    # print("====================================================================")

    #분석가능한 감정 총 감정 수 - Bert origin model 적용시 28개 감정 추출돰
    total_num_emotion_analyzed = 28

    # 감정기복 비율 계산 !!!
    result_emo_swings =round(len(unique_re)/total_num_emotion_analyzed *100,1) #소숫점 첫째자리만 표현
    # print("문장에 표현된 감정 비율 : ", result_emo_swings)
    # print("====================================================================")

    # 결과해서
    # reslult_emo_swings : 전체 문장에서의 감정 비율 계산
    # unique_re : 에세이에서 분석 추출한 감정   ====> 이것이 중요한 값임
    return result_emo_swings, unique_re





# Selected College 외 다양한 것을 계산하는 코드(최종계산코드)
# 입력값:  대학, 전공 ex) ('Why us', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
# 입력값:  대학, 전공 ex) ('Intellectual interest', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
def pmpt_orinted_sentments(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data):

    pmt_sent_etc_re = select_prompt_type(select_pmt_type)
    prompt_type_sentence = pmt_sent_etc_re[0] # prompt 문장 ex) Prompt Type : Intellectual interest 이렇게 20가지가 있음
    pmt_sent_re = list(pmt_sent_etc_re[1]) # prompt 에 해당하는 sentiment list
    intended_mjr = select_major # 희망전공

    if select_college == 'Harvard':
        pass
    elif select_college == 'Princeton':
        pass
    elif select_college == 'Stanford':
        pass
    elif select_college == 'MIT':
        pass
    elif select_college == 'Columbia':
        pass
    elif select_college == 'UPenn':
        pass
    elif select_college == 'Brown':
        College_text_data = open_data(select_college) # 선택한 대학의 정보가 담긴 txt 파일을 불러오고
        re_mjr = open_major_data(select_college, select_major) # 선택한 대학과 전공의 정보를 불러와서
        # gen_keywd_college = general_keywords(College_text_data) # 키워드 추출하여 대학정보 WordCloud로 구현
        # gen_keywd_college_major = general_keywords(re_mjr) # 키워드 추출하여 대학의 전공 WordCloud로 구현
    elif select_college == 'Cornell':
        pass
    elif select_college == 'Dartmouth':
        pass
    elif select_college == 'UChicago':
        pass
    elif select_college == 'Northwestern':
        pass
    elif select_college == 'Duke':
        pass
    elif select_college == 'Johns Hopkins':
        pass
    elif select_college == 'UCLA':
        pass
    elif select_college == 'UC Berkeley':
        pass
    elif select_college == 'Carnegie Mellon':
        pass
    elif select_college == 'Emory':
        pass
    elif select_college == 'Georgetown':
        pass
    elif select_college == 'UCLA':
        pass
    elif select_college == 'Emory':
        pass
    elif select_college == 'Caltech':
        pass
    elif select_college == 'USC':
        pass
    elif select_college == 'Georgetown':
        pass
    elif select_college == 'Willams':
        pass
    elif select_college == 'Swarthmore':
        pass
    elif select_college == 'Amherst':
        pass
    else:
        pass
    
    essay_input = coll_supp_essay_input_data

    # prompt 에 해당하는 sentiments와 관련한 감정과, 입력한 에세이에서 추출한 감정들이 얼마나 일치 비율 계산하기
    # 에세이에서 추출 분석한 감정 리스트
    get_sents_from_essay = Prompt_Oriented_Sentiments_analysis(essay_input)
    # 선택한 해당 prompt의 감정 리스트 : pmt_sent_re
    pmt_snet_re_num = len(pmt_sent_re) # 선택한 prompt에서 추출한 감정 수
    cnt = 0
    for i in pmt_sent_re:
        if i in get_sents_from_essay[1]:
            cnt += 1
    
    cnt_re = cnt # 에세이에서 추출한 감정정보와 프롬프트가 의도한(포함된) 감정정보 중 몇개가 겹치는지 카운트, 많이 겹치면 에세이가 fit 하다는 의미, 적게 겹치면 unfit

    ############ - 에세이 입력 구간을 분리하여 감성분석 시작 - ##########
    def parted_Sentiments_analysis(essay_input):
        ########## 여기서는 최초 입력 에세이를 적용한다. input_text !!!!!!!!
        re_text = essay_input.split(".")

        #데이터 전처리 
        def cleaning(datas):
            fin_datas = []
            for data in datas:
                # 영문자 이외 문자는 공백으로 변환
                only_english = re.sub('[^a-zA-Z]', ' ', data)
                # 데이터를 리스트에 추가 
                fin_datas.append(only_english)
            return fin_datas

        texts = cleaning(re_text)
        texts_40 = texts[:int(round(len(texts)*0.4,0))] # 40% 앞부분 추출
        texts_60 = texts[int(round(len(texts)*0.4,0)):] # 60%은 뒷부분 추출

        def get_emo_text_ratio(text_input_list):
            #분석된 감정만 추출
            emo_re = goemotions(texts)

            emo_all = []
            for list_val in range(0, len(emo_re)):
                #print(emo_re[list_val]['labels'],emo_re[list_val]['scores'])
                #mo_all.append((emo_re[list_val]['labels'],emo_re[list_val]['scores'])) #KEY, VALUE만 추출하여 리스트로 저장
                #emo_all.append(emo_re[list_val]['scores'])
                emo_all.append((emo_re[list_val]['labels']))
                
            #추출결과 확인 
            # emo_all

            # ['sadness'],
            #  ['anger'],
            #  ['admiration', 'realization'],
            #  ['admiration', 'disappointment'],
            #  ['love'],
            #  ['sadness', 'neutral'],
            #  ['realization', 'neutral'],
            #  ['neutral'],
            #  ['optimism'],
            #  ['neutral'],
            #  ['excitement'],
            #  ['neutral'],
            #  ['neutral'],
            #  ['caring'],
            #  ['gratitude'],
            #  ['admiration', 'approval'], ...

            from pandas.core.common import flatten #이중리스틀 FLATTEN하게 변환
            flat_list = list(flatten(emo_all))

            # ['neutral',
            #  'neutral',
            #  'sadness',
            #  'anger',
            #  'admiration',
            #  'realization',
            #  'admiration',
            #  'disappointment',


            #중립적인 감정을 제외하고, 입력한 문장에서 다양한 감정을 모두 추출하고 어떤 감정이 있는지 계산해보자
            unique = []
            for r in flat_list:
                if r == 'neutral':
                    pass
                else:
                    unique.append(r)

            #중립감정 제거 및 유일한 감정값 확인
            #unique
            unique_re = set(unique) #중복제거

            ############################################################################
            # 글에 표현된 감정이 얼마나 다양한지 분석 결과!!!¶
            # print("====================================================================")
            # print("에세이에 표현된 다양한 감정 수:", len(unique_re))
            # print("====================================================================")

            #분석가능한 감정 총 감정 수 - Bert origin model 적용시 28개 감정 추출돰
            total_num_emotion_analyzed = 28

            # 감정기복 비율 계산 !!!
            result_emo_swings =round(len(unique_re)/total_num_emotion_analyzed *100,1) #소숫점 첫째자리만 표현
            # print("문장에 표현된 감정 비율 : ", result_emo_swings)
            # print("====================================================================")

            # unique_re : 에세이에서 분석 추출한 감정   ====> 이것이 중요한 값임
            return unique_re

        result_of_each_emotion_analysis_1 = get_emo_text_ratio(texts_40)
        result_of_each_emotion_analysis_2 = get_emo_text_ratio(texts_60)

        return result_of_each_emotion_analysis_1, result_of_each_emotion_analysis_2

    ############ - 구간분리 감성분석 결과 끝 - ##############


    # 에세이 구간별 감성 분석 적용 부분 [:16], [17:]
    # pmt_sent_etc_re[:16] # 1차 감성부분 초반 40% (단어수따라 틀림), 총점의 40%: anger, annoyance, disapproval, disappointment, disgust, fear, grief, nervousness, sadness, surprise, remorse, curiosity, embarrassment
    # pmt_sent_etc_re[17:] # 2차 감성부분 후반 60% (단어수따라 틀림), 총점의 60%: admiration, approval, caring, joy, gratitude, optimism, realization, relief

    ### 문장 구간 분리하여 대표 감성 추출할 것 ###
    sent_parted_re = parted_Sentiments_analysis(essay_input)
    # 초반 40% 구간의 대표감성분석 결과
    sent_pre_40_re = sent_parted_re[0]
    # 일치비율 계산
    # 전반 40% 구간에 해당하는 감성정보값(리스트) : pmt_sent_etc_re[:16]
    # 전반 40% 구간에 해당하는 감성정보값(리스트) - overcomming chanllenge ethical dilema : anger, annoyance, confusion, embarrassment, disappointment, disapproval, fear, nervousness, sadness, remorse
    pmt_emo_40 = ['anger', 'annoyance', 'confusion', 'embarrassment', 'disappointment', 'disapproval', 'fear', 'nervousness', 'sadness', 'remorse']
    # 비교
    s_40_cnt= 0
    for ittm in sent_pre_40_re:
        if ittm in pmt_emo_40: # 전반 40% 구간에 일치하는 감성이 있다면,
            s_40_cnt += 1 # 카운트하고, 

    if s_40_cnt == 0: # 일치 하는 감성정보가 없다면,
        sent_comp_ratio_40 = 0
    else: # 있다면,
        sent_comp_ratio_40 = round(s_40_cnt / len(pmt_emo_40) * 100, 2) * 0.4 # 전반 40% 구간에서 일치하는 감성의 선택한 프롬프트감성과의 비교결과 포함 비율을 계산하고, 가중치 적용(0.4)


    # 후반 60% 구간의 대표감성분석 결과
    sent_pre_60_re = sent_parted_re[1]
    # 일치비율 계산
    # 후반 60% 구간에 해당하는 감성정보값(리스트) : pmt_sent_etc_re[17:]
    # 후반 60% 구간에 해당하는 감성정보값(리스트) - overcomming chanllenge ethical dilema : admiration, approval, caring, joy, gratitude, love, optimism, relief, pride
    pmt_emo_60 = ['admiration', 'approval', 'caring', 'joy', 'gratitude', 'love', 'optimism', 'relief', 'pride']
    s_60_cnt= 0
    for ittm_ in sent_pre_60_re:
        if ittm_ in pmt_emo_60: # 후반 60% 구간 리스트에 일치하는 감성이 있다면,
            s_60_cnt += 1 # 카운트하고, 

    if s_60_cnt == 0: # 일치 하는 감성정보가 없다면,
        sent_comp_ratio_60 = 0
    else:
        sent_comp_ratio_60 = round(s_60_cnt / len(pmt_emo_60) * 100, 2) * 0.6

    # 비율 적용한 최종 값
    fin_re_sentiments_analysis = sent_comp_ratio_40 + sent_comp_ratio_60
    #print('fin_re_sentiments_analysis:', fin_re_sentiments_analysis)

    # 일치비율 계산
    sent_comp_ratio_origin = round(cnt_re / pmt_snet_re_num * 100, 2)


    def calculate_score(sent_comp_ratio):
        if sent_comp_ratio >= 80:
            result_pmt_ori_sentiments = 'Supurb'
        elif sent_comp_ratio >= 60 and sent_comp_ratio < 80:
            result_pmt_ori_sentiments = 'Strong'
        elif sent_comp_ratio >= 40 and sent_comp_ratio < 60:
            result_pmt_ori_sentiments = 'Good'
        elif sent_comp_ratio >= 20 and sent_comp_ratio < 40:
            result_pmt_ori_sentiments = 'Mediocre'
        else: #sent_comp_ratio < 20
            result_pmt_ori_sentiments = 'Lacking'
        return result_pmt_ori_sentiments


    result_pmt_ori_sentiments= calculate_score(fin_re_sentiments_analysis) # Social issues: contribution & solution 부분의  Prompt Oriented Sentiments -- 분리적용한 부분
    #result_pmt_ori_sentiments = calculate_score(sent_comp_ratio_origin)

    return result_pmt_ori_sentiments, fin_re_sentiments_analysis
    #sentiments_anaysis: ('Good', 42.664)


# ### run ###
# essay_input = """ I inhale deeply and blow harder than I thought possible, tech/engineering pushing the tiny ember from its resting place on the candle out into the air. mit women's technology program (wtp) The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

# # 입력값:  대학, 전공 ex) ('Intellectual interest', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
# print("sentiments_anaysis:", selected_college('Intellectual interest', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input))


def lackigIdealOverboard(group_mean, personal_value): # group_mean: 1000명 평균, personal_value: 개인값
    ideal_mean = group_mean
    one_ps_char_desc = personal_value
    #최대, 최소값 기준으로 구간설정. 구간비율 30% => 0.3으로 설정
    min_ = int(ideal_mean-ideal_mean*0.6)
    #print('min_', min_)
    max_ = int(ideal_mean+ideal_mean*0.6)
    #print('max_: ', max_)
    div_ = int(((ideal_mean+ideal_mean*0.6)-(ideal_mean-ideal_mean*0.6))/3)
    #print('div_:', div_)

    #결과 판단 Lacking, Ideal, Overboard
    cal_abs = abs(ideal_mean - one_ps_char_desc) # 개인 - 단체 값의 절대값계산

    #print('cal_abs 절대값 :', cal_abs)
    compare7 = (one_ps_char_desc + ideal_mean)/6
    compare6 = (one_ps_char_desc + ideal_mean)/5
    compare5 = (one_ps_char_desc + ideal_mean)/4
    compare4 = (one_ps_char_desc + ideal_mean)/3
    compare3 = (one_ps_char_desc + ideal_mean)/2
    # print('compare7 :', compare7)
    # print('compare6 :', compare6)
    # print('compare5 :', compare5)
    # print('compare4 :', compare4)
    # print('compare3 :', compare3)

    if one_ps_char_desc > ideal_mean: # 개인점수가 평균보다 클 경우는 overboard
        if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
            #print("Overboard: 2")
            result = 2 #overboard
            score = 1
        elif cal_abs > compare4: # 28
            #print("Overvoard: 2")
            result = 2
            score = 2
        elif cal_abs > compare5: # 22
            #print("Overvoard: 2")
            result = 2
            score = 3
        elif cal_abs > compare6: # 18
            #print("Overvoard: 2")
            result = 2
            score = 4
        else:
            #print("Ideal: 1")
            result = 1
            score = 5
    elif one_ps_char_desc < ideal_mean: # 개인점수가 평균보다 작을 경우 lacking
        if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
            #print("Lacking: 2")
            result = 0
            score = 1
        elif cal_abs > compare4: # 28
            #print("Lacking: 2")
            result = 0
            score = 2
        elif cal_abs > compare5: # 22
            #print("Lacking: 2")
            result = 0
            score = 3
        elif cal_abs > compare6: # 18
            #print("Lacking: 2")
            result = 0
            score = 4
        else:
            #print("Ideal: 1")
            result = 1
            score = 5
            
    else: # 같으면 ideal 이지. 가장 높은 점수를 줄 것
        #print("Ideal: 1")
        result = 1
        score = 5

    # 최종 결과 5점 척도로 계산하기
    if score == 5:
        result_ = 'Supurb'
        re__score = 100
    elif score == 4:
        result_ = 'Strong'
        re__score = 80
    elif score == 3:
        result_ = 'Good'
        re__score = 60
    elif score == 2:
        result_ = 'Mediocre'
        re__score = 40
    else: #score = 1
        result_ = 'Lacking'
        re__score = 20

    return result_, re__score



# conflict
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

        #표현하는 단어들을 리스트에 넣어서 필터로 만들고 --  conflict 단어 업데이트 해야 함
        confict_words_list = ['clash', 'incompatible', 'inconsistent', 'incongruous', 'opposition', 'variance','vary', 'odds', 
                                'differ', 'diverge', 'disagree', 'contrast', 'collide', 'contradictory', 'incompatible', 'conflict',
                                'inconsistent','irreconcilable','incongruous','contrary','opposite','opposing','opposed',
                                'antithetical','clashing','discordant','differing','different','divergent','discrepant',
                                'varying','disagreeing','contrasting','at odds','in opposition','at variance','oyster','disputing','breeze','verbal','organization','formal','distinct','petty','blaine','schism','unrestrained','conform','friction','must','firing','played','sound','incidental','dog','agreement','make','boxing','conformity','street','character','operation','decision','chafed','strive','incongruence','widespread','attitude','property','fought','possession','harden','person','cookie','antagonistic','bring','sparked','several','falling-out','bare','skirmish','struggle','competition','disband','intricate','determine','victory','fighting','horseback','attempt','crossbow','tiff','harmonious','limited','rule','incongruousness','exposure','sometimes','fist','got','agree','awarded','breaking','variant','gently','appropriate','ship','king’s','sign','fissure','triangle','collide','bos','unattained','proceeding','reasoning','argle-bargle','division','bitter','j','meet','violator','held','fray','peace','tension','base','oar','instance','fit','mêlée','pot','continuous','another','disturbance','using','resentment','concordance','abrade','noisily','part','racetrack','usually','ruction','grate','blow','word','car','pacifying','approve','contested','long','earthenware','arrangement','among','cause','deed','raise','contour','richard','mix-up','petulant','disunity','respect','fight','cold','asked','expression','similar','bivalve','confusion','reality','declaration','statesman','antipathy','embroiling','cooperation','maintained','conflicting','royal','fisticuffs','antagonism','opinion','water','construction','brawl','entire','related','country','without','purpose','difference','visible','dissension','hostility','disarmed','boat','quarreling','demobilization','parent','crack','lack','separation','made','sustained','table','freedom','squabble','gotten','crust','spat','helpful','glass','animosity','prize','expose','one’s','embarrassing','demilitarization','characteristic','disorderly','spontaneous','backing','competing','form','vex','jabber','termination','potential','harmonize','failure','may','babel','ideological','variation','scrum','sailing','asking','race','exasperating','expansion','sense','rough-and-tumble','mason','relation','fail','trained','poem','due','confrontation','end','tree','wa','peaceful','accord','disorder','bile','discordance','two','believe','drunken','kickup','grapple','motion','suggestion','donnybrook','prevent','casino','agreed','lacking','discordancy','scrimmage','correspond','james','opponent','toward','comer','especially','dividing','involving','outsider','duration','rivalry','engage','dustup','chafe','open','earned','would','argument','impact','heated','dissention','perelman','enemy','dislike','radiant','point','shape','men','coincide','rubbing','puck','ball','neighboring','stood','grasping','jar','competitor','length','widemouthed','pottery','peevishly','peaceable','divided','winner','differ','effort','opposing','motionless','sweepstakes','free-for-all','hylton','undertaken','altercation','forward','occurring','tussle','localized','principal','defeat','discord','unfriendly','compare','final','furrow','strife','punch-out','grant','became','hatred','discussion','match','truth','horse','confused','whose','rancor','order','blew','sandpaper','misunderstanding','tangle','unite','contest','rival','divide','cease-fire','state','food','melody','allowing','enmity','draw','contend','debate','churchill','game','one','grill','fall','inharmoniousness','supremacy','typically','control','overt','argy-bargy','assent','two','tossed','common','ruckus','specified','joust','collided','song','humor','violent','nation','jangling','man-at-arms','age','hand','placing','dropped','present','bridge','could','crop','pacified','wrangle','young','short','gain','failed','words','affray','condition','waged','time','international','separate','heroic','synonyms','activity','combat','angry','disposition','advanced','controversy','something','child','ideal','variable','unity','collision','polite','alphabetical','clashed','gall','religious','positive','view','withhold','anxiety','—','occurrence','catfight','index','discordant','railroad','noise','fan','tending','chat','error','active','strong','r','le','aim','best','quality','thoughtful','oven','incompatible','extremist','helicopter','total','disarmament','method','noisy','quarter','pacification','someone','dissonance','yearly','contention','disputation','period','disagree','author','brawling','difficulty','rugby','political','diplomatic','democracy','armed','row','mutual','truce','argumentation','rough','fact','fighting','storm','process','reason','melee','tug-of-war','full','square-headed','parting','duel','lacrosse','inopportune','larger','issue','police','support','criticism','harmony','angrily','sweep-stake','disunion','citizen','attainment','discarded','equal','marked','proposal','earth’s','locked','individual','war','arguing','container','falling','give','disharmony','antonyms','action','problem','tranquillity','protracted','close','square','armistice','approval','inconsistent','wood','playing','conflict','countryside','come','scuffle','deep','knit','miff','broil','foot','cymbal','carried','away','suspension','incompatibility','world','blend','quiet','beginning','actual','battalion','politics','force','hassle','inconsonance','sing','accorded','high','conflagration','dinner','heat','tendency','case','life','civil','intended','coincides','quarrel','opening','immediate','salvadori','outline','punch-up','idea','misinterpretation','side','calm','soccer','confronted','want','tight','formation','exchange','narrow','instruction','account','bolt','heavy','vie','showdown','inharmony','nail-biter','battle','steak','considerable','tranquility','objective','engaged','dissidence','union','discussing','direct','hot','jangle','cook','peacefulness','play','coincided','matter','poker','showed','dogfight','risk','book','face-off','tax','city','near','dispute','movement','aroused','consideration','feeling','minor','pocket','patrician','disaccord','unlike','dictatorship','face','understand','intense','done','occupy','express','powerful','raging','combatant','thing','disagreeing','slope','contretemps','declared','lively','run-in','winston','principle','intensely','military','group','disagreement','wanted','interest','divorce','context','warfare','inconsistence','place','450ƍf','space','incongruous','woman','written','mix','cross','antagonist','carefully','land','still','fire','infighting','area','incongruity','concurrence','slugfest','hockey','cease','ill','faceup','fragment','public','ringing','mean','power','people','participated','consonance','strenuous','knight','placed','act','pitched','built','variance','regular','sir','concur','admit','right','key','nature','engagement','admitting','tuneful','law','mario','irritate','campaign','considered','handgrip','harsh','hostile','depth','designed','propel','physical','leftover','together','clash','differs','student','bicker','congruity','concord','work','easily','rivaling','offer','opposition','confronting','sutherland','concede','wind','struggling','often','grappling','holy','solid','inconsistency','ward','deep-seated','line','scrap','colliding','induces','fracas','body','arrow','dissent','glad','fistfight','situation','tranquil','deprive']

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
        
        print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
        
        filtered_chr_text_ = set(filtered_chr_text) #중복제거
        filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
        #print (filtered_chr_text__) # 중복값 제거 확인
        
        for i in filtered_chr_text__:
            ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
        char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 표현 수
        char_count_ = len(filtered_chr_text__) #중복제거된  표현 총 수
            
        result_char_ratio = round(char_total_count/total_words * 100, 2)

        import pandas as pd

        df_conf_words = pd.DataFrame(ext_sim_words_key, columns=['words','values']) #데이터프레임으로 변환
        df_r = df_conf_words['words'] #words 컬럼 값 추출
        ext_sim_words_key = df_r.values.tolist() # 유사단어 추출

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

    # conflict 단어 업데이트 해야 함
    confict_words_list_basic = ['clash', 'incompatible', 'inconsistent', 'incongruous', 'opposition', 'variance','vary', 'odds', 
                                'differ', 'diverge', 'disagree', 'contrast', 'collide', 'contradictory', 'incompatible', 'conflict',
                                'inconsistent','irreconcilable','incongruous','contrary','opposite','opposing','opposed',
                                'antithetical','clashing','discordant','differing','different','divergent','discrepant',
                                'varying','disagreeing','contrasting','at odds','in opposition','at variance','oyster','disputing','breeze','verbal','organization','formal','distinct','petty','blaine','schism','unrestrained','conform','friction','must','firing','played','sound','incidental','dog','agreement','make','boxing','conformity','street','character','operation','decision','chafed','strive','incongruence','widespread','attitude','property','fought','possession','harden','person','cookie','antagonistic','bring','sparked','several','falling-out','bare','skirmish','struggle','competition','disband','intricate','determine','victory','fighting','horseback','attempt','crossbow','tiff','harmonious','limited','rule','incongruousness','exposure','sometimes','fist','got','agree','awarded','breaking','variant','gently','appropriate','ship','king’s','sign','fissure','triangle','collide','bos','unattained','proceeding','reasoning','argle-bargle','division','bitter','j','meet','violator','held','fray','peace','tension','base','oar','instance','fit','mêlée','pot','continuous','another','disturbance','using','resentment','concordance','abrade','noisily','part','racetrack','usually','ruction','grate','blow','word','car','pacifying','approve','contested','long','earthenware','arrangement','among','cause','deed','raise','contour','richard','mix-up','petulant','disunity','respect','fight','cold','asked','expression','similar','bivalve','confusion','reality','declaration','statesman','antipathy','embroiling','cooperation','maintained','conflicting','royal','fisticuffs','antagonism','opinion','water','construction','brawl','entire','related','country','without','purpose','difference','visible','dissension','hostility','disarmed','boat','quarreling','demobilization','parent','crack','lack','separation','made','sustained','table','freedom','squabble','gotten','crust','spat','helpful','glass','animosity','prize','expose','one’s','embarrassing','demilitarization','characteristic','disorderly','spontaneous','backing','competing','form','vex','jabber','termination','potential','harmonize','failure','may','babel','ideological','variation','scrum','sailing','asking','race','exasperating','expansion','sense','rough-and-tumble','mason','relation','fail','trained','poem','due','confrontation','end','tree','wa','peaceful','accord','disorder','bile','discordance','two','believe','drunken','kickup','grapple','motion','suggestion','donnybrook','prevent','casino','agreed','lacking','discordancy','scrimmage','correspond','james','opponent','toward','comer','especially','dividing','involving','outsider','duration','rivalry','engage','dustup','chafe','open','earned','would','argument','impact','heated','dissention','perelman','enemy','dislike','radiant','point','shape','men','coincide','rubbing','puck','ball','neighboring','stood','grasping','jar','competitor','length','widemouthed','pottery','peevishly','peaceable','divided','winner','differ','effort','opposing','motionless','sweepstakes','free-for-all','hylton','undertaken','altercation','forward','occurring','tussle','localized','principal','defeat','discord','unfriendly','compare','final','furrow','strife','punch-out','grant','became','hatred','discussion','match','truth','horse','confused','whose','rancor','order','blew','sandpaper','misunderstanding','tangle','unite','contest','rival','divide','cease-fire','state','food','melody','allowing','enmity','draw','contend','debate','churchill','game','one','grill','fall','inharmoniousness','supremacy','typically','control','overt','argy-bargy','assent','two','tossed','common','ruckus','specified','joust','collided','song','humor','violent','nation','jangling','man-at-arms','age','hand','placing','dropped','present','bridge','could','crop','pacified','wrangle','young','short','gain','failed','words','affray','condition','waged','time','international','separate','heroic','synonyms','activity','combat','angry','disposition','advanced','controversy','something','child','ideal','variable','unity','collision','polite','alphabetical','clashed','gall','religious','positive','view','withhold','anxiety','—','occurrence','catfight','index','discordant','railroad','noise','fan','tending','chat','error','active','strong','r','le','aim','best','quality','thoughtful','oven','incompatible','extremist','helicopter','total','disarmament','method','noisy','quarter','pacification','someone','dissonance','yearly','contention','disputation','period','disagree','author','brawling','difficulty','rugby','political','diplomatic','democracy','armed','row','mutual','truce','argumentation','rough','fact','fighting','storm','process','reason','melee','tug-of-war','full','square-headed','parting','duel','lacrosse','inopportune','larger','issue','police','support','criticism','harmony','angrily','sweep-stake','disunion','citizen','attainment','discarded','equal','marked','proposal','earth’s','locked','individual','war','arguing','container','falling','give','disharmony','antonyms','action','problem','tranquillity','protracted','close','square','armistice','approval','inconsistent','wood','playing','conflict','countryside','come','scuffle','deep','knit','miff','broil','foot','cymbal','carried','away','suspension','incompatibility','world','blend','quiet','beginning','actual','battalion','politics','force','hassle','inconsonance','sing','accorded','high','conflagration','dinner','heat','tendency','case','life','civil','intended','coincides','quarrel','opening','immediate','salvadori','outline','punch-up','idea','misinterpretation','side','calm','soccer','confronted','want','tight','formation','exchange','narrow','instruction','account','bolt','heavy','vie','showdown','inharmony','nail-biter','battle','steak','considerable','tranquility','objective','engaged','dissidence','union','discussing','direct','hot','jangle','cook','peacefulness','play','coincided','matter','poker','showed','dogfight','risk','book','face-off','tax','city','near','dispute','movement','aroused','consideration','feeling','minor','pocket','patrician','disaccord','unlike','dictatorship','face','understand','intense','done','occupy','express','powerful','raging','combatant','thing','disagreeing','slope','contretemps','declared','lively','run-in','winston','principle','intensely','military','group','disagreement','wanted','interest','divorce','context','warfare','inconsistence','place','450ƍf','space','incongruous','woman','written','mix','cross','antagonist','carefully','land','still','fire','infighting','area','incongruity','concurrence','slugfest','hockey','cease','ill','faceup','fragment','public','ringing','mean','power','people','participated','consonance','strenuous','knight','placed','act','pitched','built','variance','regular','sir','concur','admit','right','key','nature','engagement','admitting','tuneful','law','mario','irritate','campaign','considered','handgrip','harsh','hostile','depth','designed','propel','physical','leftover','together','clash','differs','student','bicker','congruity','concord','work','easily','rivaling','offer','opposition','confronting','sutherland','concede','wind','struggling','often','grappling','holy','solid','inconsistency','ward','deep-seated','line','scrap','colliding','induces','fracas','body','arrow','dissent','glad','fistfight','situation','tranquil','deprive']

    confict_words_list = confict_words_list_basic + conflict_sim_words_ratio_result #유사단어를 계산결과 반영!

    count_conflict_list = []
    for i in token_list_str:
        for j in confict_words_list:
            if i == j:
                count_conflict_list.append(j)

    #len(count_conflict_list) # 한 문장에 들어있는 conflict 단어 수

    #입력한 글을 모두 단어로 쪼개로 리스트로 만들기 - 
    essay_input_corpus_ = str(input_text) #문장입력
    essay_input_corpus_ = essay_input_corpus_.lower()#소문자 변환

    sentences_  = sent_tokenize(essay_input_corpus_) #문장단위로 토큰화(구분)되어 리스에 담김

    # 문장을 토크큰화하여 해당 문장에 Action Verbs가 있는지 분석 부분 코드임 ---> 뒤에서 나옴 아래 777로 표시된 코드부분에서 sentences_ 값 재활용

    split_sentences_ = []
    for sentence in sentences_:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences_.append(words)
        
    # 입력한 문장을 모두 리스트로 변환
    input_text_list = [y for x in split_sentences_ for y in x] # 이중 리스트 Flatten

    ############# Degree of Conflict  비율 계산 #################
    conflict_word_ratio = round(len(count_conflict_list) / len(input_text_list) * 100, 1)  
    print("Degree of conflict  단어가 전체 문장(단어)에서 차지하는 비율 계산 :", conflict_word_ratio)

    #################  합격한 학생들의 컨플릭 단어 사용 평균 값 #########
    #################  합격한 학생들의 컨플릭 단어 사용 평균 값 #########
    #################  합격한 학생들의 컨플릭 단어 사용 평균 값 #########
    group_mean = 7 # 합격한 학새의 컨플릭트 사용율 보정 필요!!!!!!!
    result = lackigIdealOverboard(group_mean, conflict_word_ratio)

    return result
    #('Supurb', 100)




### run ###
# essay_input = """ I inhale deeply and blow harder than I thought possible, tech/engineering pushing the tiny ember from its resting place on the candle out into the air. mit women's technology program (wtp) The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

# print('conflict: ', ai_plot_conf(essay_input))
#('Supurb', 100)

def calculate_score(input_scofre):
    if input_scofre >= 80:
        result_comm_ori_keywordss = 'Supurb'
    elif input_scofre >= 60 and input_scofre < 80:
        result_comm_ori_keywordss = 'Strong'
    elif input_scofre >= 40 and input_scofre < 60:
        result_comm_ori_keywordss = 'Good'
    elif input_scofre >= 20 and input_scofre < 40:
        result_comm_ori_keywordss = 'Mediocre'
    else: #input_scofre < 20
        result_comm_ori_keywordss = 'Lacking'
    return result_comm_ori_keywordss

#supurb ~ lacking 을 숫자로 된 점수로 변환
def text_re_to_score(input):
    if input == 'Supurb':
        tp_knowledge_re = 90
    elif input == 'Strong':
        tp_knowledge_re = 75
    elif input == 'Good':
        tp_knowledge_re = 65
    elif input == 'Mediocre':
        tp_knowledge_re = 40
    else: #input == 'Lacking'
        tp_knowledge_re = 10
    return tp_knowledge_re


# initive engagement
def engagement(essay_input):

    Engagement_result = intellectualEnguagement(essay_input)
    Engagement_result_fin = calculate_score(Engagement_result[0])
    engagement_re_final = text_re_to_score(Engagement_result_fin)
    
    return engagement_re_final, Engagement_result[2]


### 종합계산(이것을 실행하면 됨) ###
def overcom_chall_ethi_dilemma(select_pmt_type, select_college, select_college_dept, select_major, essay_input):
    pmpt_orinted_sentments_re = pmpt_orinted_sentments(select_pmt_type, select_college, select_college_dept, select_major, essay_input)

    conflict_re = ai_plot_conf(essay_input)

    initive_engagement = engagement(essay_input)
    # print('initive_engagement:', initive_engagement)

    engagement_result_ = calculate_score(initive_engagement[0])

    overallScore = pmpt_orinted_sentments_re[1] * 0.5 + conflict_re[1] * 0.25 + initive_engagement[0] * 0.25
    overallScore_result = calculate_score(overallScore)



      # 문장생성
    fixed_top_comment = """A challenge or dilemma in your life can be a complex mix of sentiments. Usually, it starts with negative sentiments, as this type of uncertainty can stress you out. However, since the prompt intends to see how you’ve overcome such a challenge, the essay should end on a positive note with the lessons you’ve gained from the experience."""

    def gen_comment(input_score, type):
        if input_score == 'Supurb' or input_score == 'Strong':
            if type == 'pmt_ori_sentiment':
                comment_achieve = """Your essay seems to demonstrate a superb balance between the positive and negative sentiments that shows you have successfully overcome a difficulty."""
            elif type == 'degree_of_conflict':
                comment_achieve = """Your essay seems to have a sufficient level of conflict, while"""
            elif type == 'engagement':
                comment_achieve = """the actions for overcoming the difficulty are described in great detail."""
            else:
                pass
        elif input_score == 'Good':
            if type == 'pmt_ori_sentiment':
                comment_achieve = """Your essay seems to demonstrate a satisfactory balance between the positive and negative sentiments that show you have overcome a difficulty."""
            elif type == 'degree_of_conflict':
                comment_achieve = """Your essay seems to have a satisfactory level of conflict, while"""
            elif type == 'engagement':
                comment_achieve = """the actions for overcoming the difficulty are described in detail."""
            else:
                pass
        else: #input score == 'Mediocre' or input_score == 'Weak'
            if type == 'pmt_ori_sentiment':
                comment_achieve = """Your essay may lack the balance between the positive and negative sentiments that show you have overcome a difficulty."""
            elif type == 'degree_of_conflict':
                comment_achieve = """Your essay seems to be lacking conflicts, while"""
            elif type == 'engagement':
                comment_achieve = """you may need to build upon your actions of overcoming the problem."""
            else:
                pass
        return comment_achieve


    comment_1 = gen_comment(pmpt_orinted_sentments_re[0], 'pmt_ori_sentiment')
    comment_2 = gen_comment(conflict_re[0], 'degree_of_conflict')
    comment_3 = gen_comment(engagement_result_, 'engagement')


    data = {
        'overallScore' : overallScore,
        'overallScore_result' : overallScore_result,
        'pmpt_orinted_sentments_re[0]' : pmpt_orinted_sentments_re[0], # Supurb ~ Lacking
        'pmpt_orinted_sentments_re[1]' : pmpt_orinted_sentments_re[1], # score(number)
        'conflict_re[0]' : conflict_re[0], # # Supurb ~ Lacking
        'conflict_re[1]' : conflict_re[1], # # score(number)
        'engagement_result_' : engagement_result_, # Supurb ~ Lacking
        'initive_engagement[0]' : initive_engagement[0], # score(number)
        'initive_engagement[1]' : initive_engagement[1], # 관련 단어 추출

        'fixed_top_comment': fixed_top_comment,
        'comment_1' : comment_1,
        'comment_2' : comment_2,
        'comment_3' : comment_3,
    }

    return data




essay_input = """ I inhale deeply and blow harder than I thought possible, tech/engineering pushing the tiny ember from its resting place on the candle out into the air. mit women's technology program (wtp) The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

print(overcom_chall_ethi_dilemma('Intellectual interest', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input))
