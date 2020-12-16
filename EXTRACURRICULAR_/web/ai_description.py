# 가장 Leadership+Dedication+Major Fit 점수 합계가 높은 상위 60% 활동만 계산에 활용 
# (각 활동별로 Dedication level (60%) + Leadership level (40%) + Major Fit 가산점 더해서 가장 높은 순위부터 계산에 활용… 중요한 활동부터)

# dedicaton 값은 leadership, major_activity 값과 합쳐서 한개의 함수로 처리하여 계산할 것

#### 한개의 함수로 처리하여 계산!!!!!!!!!


import numpy as np
import gensim
import nltk
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

from ai_leadership import leadership_start_here, total_act_lists
from ai_major_activity_fit  import mjr_act_analy, total_actvity, each_mjr_act_fit_analysis
from ai_dedication_analysis  import dedication_analysis


######################## 웹사이트에서 값을 입력받아야 함 ##################################### - start -

#input
hrs_per_week = 5 #time_spent_hrs_per_week
weeks_per_year = 20 #tiem_spent_weeks_per_year
period_years_of_activity = 3 #period_years_of_activity

# 3개의 입력: 전공 3개
majors = """ mechanical engineering, Film Studies, Psychology  """

# 일단 3개의 EXTRACURRICULAR ACTIVITY EXAMPLES  입력, 추가로 활동을 입력할 수 있음. 최대 10개, 그 이상도 가능하지만 비율로 게산
input_text_1 = """ deputy Member (9th/10th) Treasurer (11th/12th) National Honors Society, Ridgefield High School Chapter
We are amongst the highest academically achieving students at our school, who collectively and consistently participate in community service projects.""" # 실제 값은 문장이 입력되어야 함, 현재는 테스트용 단어입력

input_text_2 = """Manager/Administrator (Summer 2019)
ViolinMan Resource, secondhand store for renting used musical instruments	
Approved and updated online information database on classical music instruments for use by music enthusiasts. Cleaned and distributed instruments for use."""

input_text_3 = """  planned and hosted team banquet; led team warmups and meetings; Coach's Award Recipient (11th); Team Spirit Award (12th).  """

input_text_4 = """ Peer Advisor (11th-12th)
Erving High School Student Ambassador Program, selective application-based leadership team
Organized and led orientation; served as a year round leader, mentor, tutor, and friend to freshmen; helped with class scheduling."""

input_text_5 = """  Leader/Concertmaster (10th-12th)
AMAC Youth Chamber Ensemble (AYCE), audition-based community choir 
Lead ensemble in rehearsal and performance, coordinate rehearsal times, aid younger   """

input_text_6 = """ researched teaching pedagogy, provided positive feedback to encourage kids.  """

input_text_7 = """ 
8 months intensive preparation on English language proficiency and presentation skills for British English Olympics. Won 6th place out of 50 schools. """

input_text_8 = """Student Coach (9th - 12th)
Middle School MathCounts Team
Taught strategies, selected competitors, hosted weekly practice sessions and lectures. Led team to 2nd place victory at State Mathematics competition (11th). """

input_text_9 = """ Protein Modeling Team Leader (10th)
Science Olympiad, Burke High School Club
Supervised building of protein molecule model, taught peers to use 3D molecular program Jmol; placed in top ten in 2017 regional competition. """

input_text_10 = """""" #이것은 값이 없기 때문에 null로 처리해 보자


## 활동입력 값 리스트에 담기
total_actvity = [input_text_1, input_text_2, input_text_3, input_text_4, input_text_5, input_text_6,input_text_7, input_text_9, input_text_10]

total_activity_num = len(total_actvity)
#print'(total_activity_num :' total_activity_num)



#############################################################################################################

# 캐릭터(문자) 수 계산
def character_counter(text):

    input_corpus = str(text) #문장입력
    input_corpus = input_corpus.lower()#소문자 변환

    sentences  = sent_tokenize(input_corpus) #문장 토큰화
    total_sentences = len(sentences)#토큰으로 처리된 총 문장 수
    total_words = len(word_tokenize(input_corpus))# 총 단어수
    split_sentences = []
    for sentence in sentences:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences.append(words)
    
    re__ = sum(split_sentences,[])
    result = len(re__)
        
    return result

# 입력 활동별 점수 계산
def character_counter_scoring(total_actvity):
    result_chr_count_ = [] # 문자수 계산
    for i in total_actvity:
        result_chr_count = character_counter(i)
        print("개별항목 입력문자수:" , result_chr_count)
        result_each_char = round((result_chr_count / 36)*100,2)
        result_chr_count_.append(result_each_char)

        ##### 조건문으로 입력수에 대한 전체 평균값을 계산해야 함... description  강도 계산하기~!!!!!!!

    return result_chr_count_


####### 입력 활동별 점수 계산 실행 !
each_desc_score = character_counter_scoring(total_actvity)
print ('입력활동 개별 점수 계산 : ', each_desc_score)


#############################################################################################################

# action verbs 수 계산
def action_verbs_counter(text):
    
    #csv 파일에서 단어 사전 불러오기
    data_action_verbs = pd.read_csv('actionverbs.csv')
    data_ac_verbs_list = data_action_verbs.values.tolist()
    verbs_list = [y for x in data_ac_verbs_list for y in x]
    #print(verbs_list)
    
    # 입력문장 처리
    input_corpus = str(text) #문장입력
    input_corpus = input_corpus.lower()#소문자 변환

    sentences  = sent_tokenize(input_corpus) #문장 토큰화
    total_sentences = len(sentences)#토큰으로 처리된 총 문장 수
    total_words = len(word_tokenize(input_corpus))# 총 단어수
    split_sentences = []
    for sentence in sentences:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences.append(words)
    
    re_act_verbs = sum(split_sentences,[])
    
    ext_re = []
    for actv_item in verbs_list:
        for sent_item in re_act_verbs:
            if actv_item  == sent_item:
                ext_re.append(actv_item)
    
    return ext_re




#### 전공 + 특별활동 >> 개별항목에 대한 계산결과를 도출해야 함

# def each_mjr_act_fit_analysis(majors, total_actvity):
#     each_m_a_result = []
#     for i in total_actvity:
#         result_each_mjr_act = mjr_act_analy(majors, i)
#         each_m_a_result.append(result_each_mjr_act)
#     return each_m_a_result


re_each_M_A = each_mjr_act_fit_analysis(majors, total_actvity)
print("==================================")
print('RESULT major - activity fit :', re_each_M_A)

#  ==== 결과 예시 ====  #
 
# RESULT major - activity fit : (['5', '5', '1', '1', '1', '1', '1', '1', '1'], 1.89, [0, 1, 2, 3, 4, 5])

#  ==== 결과 해석 ====  #

# 개별항목 계산 결과 : each_m_a_result >>>   ['5', '5', '1', '1', '1', '1', '1', '1', '1']
# 전체평균 : all_m_a_result >>>>>>>>>>>>>    1.89
# 상위 6개의 입력값 위치(우수한 활동내역 순서대로 추출) : re_top6 >>>> [0, 1, 2, 3, 4, 5]




result_leadership_fin = leadership_start_here(total_act_lists)
print ("=================================")
print ('RESULT leadership :', result_leadership_fin)


result = dedication_analysis(hrs_per_week, weeks_per_year, period_years_of_activity)
print ("=================================")
print ("RESULT dedecation :", result)
print ("=================================")
