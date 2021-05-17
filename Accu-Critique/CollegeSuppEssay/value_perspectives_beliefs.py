# Prompt Type: Values, perspectives, or beliefs

# Prompt Oriented Keywords
import nltk
import re
import numpy as np
import pandas as pd
import gensim
from nltk.tokenize import sent_tokenize
import multiprocessing
import io
from gensim.models import Phrases
from textblob import TextBlob
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop = stopwords.words('english')

from intellectualEngagement import intellectualEnguagement


### Prompt Oriented Keywords ###
# 토픽 추출
from prompt_oriented_keywords import prompt_oriented_key_wds
from perspective import getPerspectiveWD


# Prompt Oriented Sentiments  -- 글속에 감정이 얼마나 표현되어 있는지 분석 - origin (Bert pre trained model 활용)
import numpy as np
import spacy
from collections import Counter
import re
import nltk
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load("en_core_web_sm")


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

##   이것을 실행할 것!
def pmp_ori_kwds(prompt_type, essay_input):
    topic_result = prompt_oriented_key_wds(prompt_type, essay_input)
    topic_ext_result = topic_result[0] #  Prompt Oriented Keywords 값 계산 결과 Supurb ~ Lacking
    topic_score = text_re_to_score(topic_ext_result)
    disp_web = topic_result[2] # Prompt Oriented Keywords 단어들(웹사이트에 표시)

    # 0. topic_ext_result : Supurb ~ Lacking
    # 1. topic_score : numeric number
    # 2. disp_web : words list

    return topic_ext_result, topic_score, disp_web


#################################################################
# prompt oriented sentiments anaysis #
#################################################################

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
        pmt_typ = ["Culture & diversity"]
        pmt_sentiment = ['Admiration','Realization','Love','Approval','Pride','Gratitude']
    elif prompt_type ==  'Collaboration & teamwork':
        pmt_typ = ["Collaboration & teamwork"]
        pmt_sentiment = ['Admiration','Caring','Approval','Optimism','Gratitude','Love']
    elif prompt_type ==  'Creativity/creative projects':
        pmt_typ = ["Creativity/creative projects"]
        pmt_sentiment = ['Excitement','Realization','Curiosity','Desire','Amusement','Surprise']
    elif prompt_type ==  'Leadership experience':
        pmt_typ = ["Leadership experience"]
        pmt_sentiment = ['Admiration','Caring','Approval','Optimism','Gratitude','Love','Fear','Confusion','Nervousness']
    elif prompt_type ==  'Values, perspectives, or beliefs':
        pmt_typ = ["Values, perspectives, or beliefs"]
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
    #pmt_emo_40 = ['anger', 'annoyance', 'confusion', 'embarrassment', 'disappointment', 'disapproval', 'fear', 'nervousness', 'sadness', 'remorse']
    # 비교
    s_40_cnt= 0
    for ittm in sent_pre_40_re:
        if ittm in pmt_sent_etc_re[:16]: # 전반 40% 구간에 일치하는 감성이 있다면,
            s_40_cnt += 1 # 카운트하고, 

    if s_40_cnt == 0: # 일치 하는 감성정보가 없다면,
        sent_comp_ratio_40 = 0
    else: # 있다면,
        sent_comp_ratio_40 = round(s_40_cnt / len(pmt_sent_etc_re[:16]) * 100, 2) * 0.4 # 전반 40% 구간에서 일치하는 감성의 선택한 프롬프트감성과의 비교결과 포함 비율을 계산하고, 가중치 적용(0.4)


    # 후반 60% 구간의 대표감성분석 결과
    sent_pre_60_re = sent_parted_re[1]
    # 일치비율 계산
    # 후반 60% 구간에 해당하는 감성정보값(리스트) : pmt_sent_etc_re[17:]
    # 후반 60% 구간에 해당하는 감성정보값(리스트) - overcomming chanllenge ethical dilema : admiration, approval, caring, joy, gratitude, love, optimism, relief, pride
    #pmt_emo_60 = ['admiration', 'approval', 'caring', 'joy', 'gratitude', 'love', 'optimism', 'relief', 'pride']
    s_60_cnt= 0
    for ittm_ in sent_pre_60_re:
        if ittm_ in pmt_sent_etc_re[17:]: # 후반 60% 구간 리스트에 일치하는 감성이 있다면,
            s_60_cnt += 1 # 카운트하고, 

    if s_60_cnt == 0: # 일치 하는 감성정보가 없다면,
        sent_comp_ratio_60 = 0
    else:
        sent_comp_ratio_60 = round(s_60_cnt / len(pmt_sent_etc_re[17:]) * 100, 2) * 0.6

    # 비율 적용한 최종 값
    fin_re_sentiments_analysis = sent_comp_ratio_40 + sent_comp_ratio_60
    #print('fin_re_sentiments_analysis:', fin_re_sentiments_analysis)

    # 일치비율 계산
    sent_comp_ratio_origin = round(cnt_re / pmt_snet_re_num * 100, 2)


    def calculate_score(sent_comp_ratio):
        if sent_comp_ratio >= 80:
            result_topic_uniquenesss = 'Supurb'
        elif sent_comp_ratio >= 60 and sent_comp_ratio < 80:
            result_topic_uniquenesss = 'Strong'
        elif sent_comp_ratio >= 40 and sent_comp_ratio < 60:
            result_topic_uniquenesss = 'Good'
        elif sent_comp_ratio >= 20 and sent_comp_ratio < 40:
            result_topic_uniquenesss = 'Mediocre'
        else: #sent_comp_ratio < 20
            result_topic_uniquenesss = 'Lacking'
        return result_topic_uniquenesss


    result_topic_uniquenesss= calculate_score(fin_re_sentiments_analysis) # Social issues: contribution & solution 부분의  Prompt Oriented Sentiments -- 분리적용한 부분
    # result_topic_uniquenesss = calculate_score(sent_comp_ratio_origin)
    # print('sent_pre_60_re :' , list(sent_pre_60_re))
    # print('sent_pre_40_re :' , list(sent_pre_40_re))
    all_sent = []
    for sent_itm in sent_parted_re:
        sent_ext_all = list(sent_itm)
        all_sent.append(sent_ext_all)
    
    print('all_sent:', all_sent)




    return result_topic_uniquenesss, fin_re_sentiments_analysis, all_sent
    # sentiments_anaysis: ('Lacking', 0)



### run ###
# essay_input = """ I inhale deeply and blow harder than I thought possible, tech/engineering pushing the tiny ember from its resting place on the candle out into the air. mit women's technology program (wtp) The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

# # 입력값:  대학, 전공 ex) ('Intellectual interest', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
# print("sentiments_anaysis:", pmpt_orinted_sentments("Collaboration & teamwork", "Brown", "Brown_African Studies_dept", "African Studies", essay_input))



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

########################################
# Engagememt #
def engagement(essay_input):

    Engagement_result = intellectualEnguagement(essay_input)
    engagement_words_list = Engagement_result[2]
    Engagement_result_fin = calculate_score(Engagement_result[0])
    engagement_re_final = text_re_to_score(Engagement_result_fin)
    
    return engagement_re_final, Engagement_result[3], engagement_words_list


### 이것을 실행 하시오 ###
def value_perspective_beliefs(prompt_type, select_college, select_college_dept, select_major, essay_input):


    # Keywords
    prompt_ori_keywords_re = pmp_ori_kwds(prompt_type, essay_input)
    topic_ext_result = prompt_ori_keywords_re[0] # 0. topic_ext_result : Supurb ~ Lacking
    topic_score = prompt_ori_keywords_re[1] # 1. topic_score : numeric number --> overall score 계산 반영
    print('topic_score :', topic_score)
    prompt_ori_keywords_disp_web = prompt_ori_keywords_re[2] # 2. disp_web : words list

    # Sentiments
    prompt_ori_sentiments_result = pmpt_orinted_sentments(prompt_type, select_college, select_college_dept, select_major, essay_input)
    prompt_ori_sentiments_result_5div = prompt_ori_sentiments_result[0]
    prompt_ori_sentiments_result_numeric_score = prompt_ori_sentiments_result[1] # --> overall score 계산 반영
    print('prompt_ori_sentiments_result_numeric_score:', prompt_ori_sentiments_result_numeric_score)
    extracted_total_emotional_exp_list = prompt_ori_sentiments_result[2] # --> web에 표시할 추출감성 단어들

    # Engagememt #
    eagagement_result = engagement(essay_input)
    engagement_re_final = eagagement_result[0] # Supurb ~ Lacking
    Engagement_result = eagagement_result[1] # numeric score ---> overall
    print('Engagement_result :', Engagement_result)
    engagement_words_lists_re = eagagement_result[2] # 단어들 --> web에 표시


    PS = getPerspectiveWD(essay_input)
    PS_re = PS[0] # score numeric
    print('PS:', PS_re)
    PS_5Div_score = calculate_score(PS_re)

    overallScore = round(topic_score * 0.2 + prompt_ori_sentiments_result_numeric_score * 0.3 + Engagement_result * 0.3 + PS_re * 0.2, 1)
    overallScore_5div = calculate_score(overallScore)


    # 문장생성
    fixed_top_comment = """While everyone’s values, perspectives, or beliefs may vary, it’s safe to assume that the college is looking for core values that constitute a good person. To make the essay more interesting, you may consider including the struggles and dilemmas during the process of solidifying the positive values you honor. A vivid description of the learning process and clearly stating the values you believe in would make the essay more engaging for the readers."""
    def gen_comment(input_score, type):
        if input_score == 'Supurb' or input_score == 'Strong':
            if type == 'prompt_ori_keywords':
                comment_achieve = """The value(s), perspective(s), or belief(s) you define in the essay seem to be nicely aligned with the set of core values."""
            elif type == 'prompt_ori_sentiment':
                comment_achieve = """Also, your essay seems to show a balance of positive and negative sentiments that fit the prompt's intentions perfectly."""
            elif type == 'engagement':
                comment_achieve = """All in all, your story seems to demonstrate a high level of effort and dedication for what you believe in."""
            elif type == 'perspective':
                comment_achieve = """Lastly, your perspective on the meaning and lesson seems very clear, considering your words of emphasis in the essay."""
            else:
                pass
        elif input_score == 'Good':
            if type == 'prompt_ori_keywords':
                comment_achieve = """The value(s), perspective(s), or belief(s) you define in the essay seem to be in line with the set of core values."""
            elif type == 'prompt_ori_sentiment':
                comment_achieve = """Also, your essay seems to show a balance of positive and negative sentiments that fit the prompt's intentions."""
            elif type == 'engagement':
                comment_achieve = """All in all, your story seems to demonstrate a satisfactory level of effort and dedication for what you believe in."""
            elif type == 'perspective':
                comment_achieve = """Lastly, your perspective on the meaning and lesson seems clear enough, considering your words of emphasis in the essay."""
            else:
                pass
        else: #input score == 'Mediocre' or input_score == 'Weak'
            if type == 'prompt_ori_keywords':
                comment_achieve = """The value(s), perspective(s), or belief(s) you define in the essay do not seem to match well with the set of core values."""
            elif type == 'prompt_ori_sentiment':
                comment_achieve = """Your essay may need to include more positive and negative sentiments required for the process of shaping one's values, perspective, or beliefs."""
            elif type == 'engagement':
                comment_achieve = """All in all, you may need to demonstrate a higher level of effort and dedication for what you believe in."""
            elif type == 'perspective':
                comment_achieve = """Lastly, you may consider elaborating further on your beliefs and opinions to solidify your viewpoint further."""
            else:
                pass
        return comment_achieve


    comment_1 = gen_comment(topic_ext_result, 'prompt_ori_keywords')
    comment_2 = gen_comment(prompt_ori_sentiments_result_5div, 'prompt_ori_sentiment')
    comment_3 = gen_comment(engagement_re_final, 'engagement')
    comment_4 = gen_comment(PS_re, 'perspective')    


    data = {
        'overallScore' : overallScore, 
        'overallScore_5div' : overallScore_5div, # Supurb ~ Lacking
        'topic_ext_result' : topic_ext_result, # Supurb ~ Lacking
        'prompt_ori_keywords_disp_web' : prompt_ori_keywords_disp_web, # prompt oriented keywords for web --> 웹사이트에 표시
        'prompt_ori_sentiments_result_5div' : prompt_ori_sentiments_result_5div, # Supurb ~ Lacking
        'extracted_total_emotional_exp_list' : extracted_total_emotional_exp_list, # web에 표시할 추출감성 단어들
        'engagement_re_final' : engagement_re_final, # Supurb ~ Lacking
        'engagement_words_lists_re' : engagement_words_lists_re, # web에 표시할 engagement 단어들
        'PS_re' : PS_re, # Perspective numeric score
        'PS_5Div_score' : PS_5Div_score, # Supurb ~ Lacking
        'fixed_top_comment' : fixed_top_comment,
        'comment_1' : comment_1,
        'comment_2' : comment_2,
        'comment_3' : comment_3,
        'comment_4' : comment_4,
    }

    return data 


### run ###
essay_input = """ I inhale deeply and blow harder than I thought possible, tech/engineering pushing the tiny ember from its resting place on the candle out into the air. mit women's technology program (wtp) The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""
prompt_type = 'Values, perspectives, or beliefs'

# ('Intellectual interest', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
print(value_perspective_beliefs(prompt_type, 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input))