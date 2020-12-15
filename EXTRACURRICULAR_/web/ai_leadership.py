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


# 일단 3개의 EXTRACURRICULAR ACTIVITY EXAMPLES  입력, 추가로 활동을 입력할 수 있음. 최대 10개, 그 이상도 가능하지만 비율로 게산

input_text_1 = """ deputy Member (9th/10th) Treasurer (11th/12th) National Honors Society, Ridgefield High School Chapter
We are amongst the highest academically achieving students at our school, who collectively and consistently participate in community service projects.""" # 실제 값은 문장이 입력되어야 함, 현재는 테스트용 단어입력

input_text_2 = """ Leader/Concertmaster (10th-12th)
AMAC Youth Chamber Ensemble (AYCE), audition-based community choir 
Lead ensemble in rehearsal and performance, coordinate rehearsal times, aid younger  """

input_text_3 = """ Number 1 Doubles Starter (9th-12th), Captain (11th-12th)
JV/V Beverly Hills High School Tennis Team
Three year League Champions; planned and hosted team banquet; led team warmups and meetings; Coach's Award Recipient (11th); Team Spirit Award (12th).  """

input_text_4 = """ Peer Advisor (11th-12th)
Erving High School Student Ambassador Program, selective application-based leadership team
Organized and led orientation; served as a year round leader, mentor, tutor, and friend to freshmen; helped with class scheduling."""

input_text_5 = """ Leader (11th)
Books on Global Health Equity and Social Justice, advocacy-focused peer discussion group
Researched global health equity/social justice , assigned weekly readings for group discussion, brainstormed questions to generate input from members.  """

input_text_6 = """ Number 1 Doubles Starter (9th-12th), Captain (11th-12th)
JV/V Beverly Hills High School Tennis Team
Three year League Champions; planned and hosted team banquet; led team warmups and meetings; Coach's Award Recipient (11th); Team Spirit Award (12th).  """

input_text_7 = """ Number 1 Doubles Starter (9th-12th), Captain (11th-12th)
JV/V Beverly Hills High School Tennis Team
Three year League Champions; planned and hosted team banquet; led team warmups and meetings; Coach's Award Recipient (11th); Team Spirit Award (12th).  """

input_text_8 = """Student Coach (9th - 12th)
Middle School MathCounts Team
Taught strategies, selected competitors, hosted weekly practice sessions and lectures. Led team to 2nd place victory at State Mathematics competition (11th). """

input_text_9 = """ Protein Modeling Team Leader (10th)
Science Olympiad, Burke High School Club
Supervised building of protein molecule model, taught peers to use 3D molecular program Jmol; placed in top ten in 2017 regional competition. """

input_text_10 = """""" #이것은 값이 없기 때문에 null로 처리해 보자

# 전체 활동 수 계산(입력한 활동수가 자동으로 카운트 되도록 = 여기서는 기본값으로 3개를 넣어봄)

############################################
# 웹에서 숫자가 입력되어야 함(입력한 활동 수)#  종필 이 부분 웹에서 처리하는 거야~
############################################

total_act_lists = [input_text_1, input_text_2, input_text_3, input_text_4, input_text_5,
                          input_text_6,input_text_7, input_text_8, input_text_9, input_text_10]


########### 실행함수는 맨 아래에 있음 ###############




def leadership_analysis(text):
    
    #fit_anaysis_result_fin =[]
    
    if text:
        input_corpus = str(text) #문장입력
        input_corpus = input_corpus.lower()#소문자 변환

        sentences  = sent_tokenize(input_corpus) #문장 토큰화

        split_sentences = []
        for sentence in sentences:
            processed = re.sub("[^a-zA-Z]"," ", sentence)
            words = processed.split()
            split_sentences.append(words)

        comp_txt = sum(split_sentences, [])

        superb_list = ['founder', 'co-founder', 'cofounder', 'chair', 'president', 'head', 
                   'chief', 'first author', 'captain', 'committee head', 'head of board', 
                   'board chair', 'chairperson', 'chairman', 'leader', 'CEO','ceo', 'organizer', 
                   'director', 'author', 'co-author', 'coauthor', 'chief editor', 'editor in chief']

        strong_list = ['founding member', 'VP', 'vp', 'vice Chair', 'vice', 'deputy', 'second author', 'vice captain',
                       'committee member', 'board member', 'second author', 'editor']
        good_list = ['proctor', 'prefect', 'mediator', 'third author'] # 데이터 추가할 필요가 있음

        mediocre_list = ['member', 'participant', 'helper', 'assistant']

        weak = ['Not Applicable']

        leadership_score = []

        for word in comp_txt:
            if word in superb_list:
                #print("superb")
                leadership_score.append("5")

            elif word in strong_list:
                #print("strong")
                leadership_score.append("4")

            elif word in strong_list:
                #print("good")
                leadership_score.append("3")   

            elif word in mediocre_list:
                #print("mediocre")
                leadership_score.append("2") 
            elif word == 'notapplicable': # 이게 입력데이터 선택되었을 경우 (웹에서 선택할 수 있도록 구현해야 함! not applicable > notapplicable)
                #print("Not Applicable")
                leadership_score.append("Not Applicable") 
            else:
                #print("Not Sure")
                leadership_score.append("2")

        df_fit_re = pd.DataFrame(leadership_score)
        df_fit_re.columns = ['score']
        list_fit_re = df_fit_re.drop_duplicates() #중복값 제거!!!! 결과 도출

        #조건문을 만들어서 결과를 비교 출력해보자.
        fit_anaysis_result_fin =[]
        if '5' in list_fit_re.values: # 5이 하나라도 있다면, 5 출력
            #print("5")
            fit_anaysis_result_fin.append('5')
        elif '4' in list_fit_re.values: # 4 이 있다면 , 4 출력
            #print("4")
            fit_anaysis_result_fin.append('4')
        elif '3' in list_fit_re.values: # 3 이 있다면 , 3 출력
            #print("3")
            fit_anaysis_result_fin.append('3')
        elif '2' in list_fit_re.values: # 2 이 있다면 , 2 출력
            #print("2")
            fit_anaysis_result_fin.append('2')
        elif '1' in list_fit_re.values: # 1 이 있다면 , 1 출력
            #print("1")
            fit_anaysis_result_fin.append('1')
        elif 'Not Applicable' in list_fit_re.values: # Not Applicable 이 있다면 , Not Applicable 출력
            #print("Not Applicable")
            fit_anaysis_result_fin.append('0')
            
        else:
            #print("NOT SURE")
            fit_anaysis_result_fin.append('0') #N : NOT SURE
            
        
    else:
        fit_anaysis_result_fin= '0' #N : NOT SURE
        
       

    return fit_anaysis_result_fin

# # 이 함수를 실행하면 3개의 입력값이 비교되어 계산됨
# def start_leadership_analysis(a, b, c):
#     input_act_text = a # 실제 값은 문장이 입력되어야 함, 현재는 테스트용 단어입력
#     input_act_text_ = b
#     input_act_text__ = c
#     input_txts = [input_act_text,input_act_text_, input_act_text__]
    
#     result = []
#     for i in input_txts:
#         each_re = leadership_analysis(i)
#         result.append(each_re)

#     result_leadership = sum(result, [])
#     fin_re = round((float(result_leadership[0])
#                + float(result_leadership[1])
#                + float(result_leadership[2]))/3, 2)

#     return fin_re


# ####### 실행 테스트 ####### 잘됨!!!

# re_leader = start_leadership_analysis(input_text,input_text_,input_text__) 

# #print ("=================================")
# #print ('RESULT :', re_leader)
# #print ("=================================")





############ 이것이 진짜 실행함수임  1~10개가 입력되어도 계산됨 ###############
def leadership_start_here(total_act_lists):
   
    result = []
    for i in  total_act_lists:
        each_re = leadership_analysis(i)
        result.append(each_re)

    result_ = [element for array in result for element in array]
    #문자를 숫자로 변환
    re = list(map(float,result_))

    #최종 점수(평균)
    avg = sum(re)/len(re)

    #상위 6의 입력값의 위치를 찾았음
    re_top6 = sorted(range(len(re)), key=lambda i: re[i], reverse=True)[:6]

    return avg, re_top6


############ 실행테스트!!!! ############

result_leadership_fin = leadership_start_here(total_act_lists)
#print ("=================================")
#print ('RESULT :', result_leadership_fin)
#print ("=================================")



##### 최종 계산 결과값은 4.2, 가장 높은 점수를 받은 활동은 입력한 순서대로  [1, 2, 3, 4, 5, 6] 임!
# =================================
# RESULT : (4.1, [1, 2, 3, 4, 5, 6])
# =================================