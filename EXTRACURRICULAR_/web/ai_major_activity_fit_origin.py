#이 코드는 역검증 - 주피터노트북으로 라인바이 라인 체크 필요함. 다양한 데이터로 검증 필요!


####### Major 3개 입력, 특별활동 내용 10개까지 입력한 결과를 분석하여 'Major Fit Level'을 계산하는 코드임 #####

#### 결과 ####
# superb: 5
# ===================================================
# RESULT : 5
# ===================================================




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


# 3개의 입력: 전공 3개
majors = """ mechanical engineering, Film Studies, Psychology  """



# 일단 3개의 EXTRACURRICULAR ACTIVITY EXAMPLES  입력, 추가로 활동을 입력할 수 있음. 최대 10개, 그 이상도 가능하지만 비율로 게산
input_text_1 = """""" # 실제 값은 문장이 입력되어야 함, 현재는 테스트용 단어입력

input_text_2 = """ Leader/Concertmaster (10th-12th)
AMAC Youth Chamber Ensemble (AYCE), audition-based community choir 
Lead ensemble in rehearsal and performance, coordinate rehearsal times, aid younger  """

input_text_3 = """"""

input_text_4 = """"""

input_text_5 = """"""

input_text_6 = """"""

input_text_7 = """"""

input_text_8 = """"""

input_text_9 =  """"""

input_text_10 = """""" #이것은 값이 없기 때문에 null로 처리해 보자


## 활동입력 값 리스트에 담기
total_actvity = [input_text_1, input_text_2, input_text_3, input_text_4, input_text_5, input_text_6,input_text_7, input_text_9, input_text_10]

total_activity_num = len(total_actvity)
#print'(total_activity_num :' total_activity_num)

# 활동내역을 모두 토큰화해서 리스트로 담는 코드
def tokenized(text):

    input_corpus = str(text) #문장입력
    input_corpus = input_corpus.lower()#소문자 변환

    sentences  = sent_tokenize(input_corpus) #문장 토큰화

    split_sentences = []
    for sentence in sentences:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences.append(words)
    result = sum(split_sentences, [])
    return result


def input_majors(major_txt, act_txt_list):
    ## 1. 전공입력처리 부분 ##
    major_list_ = major_txt.split(",")
    #소문자로 변환, 콤마로 구분하여 리스트로 변환 
    
    major_list = [] # 예) ['mechanical engineering', 'film studies', 'psychology']
    
    for mjl in major_list_:
        mjl_ = mjl.lower()
        mjl__ = mjl_.strip()
        major_list.append(mjl__)
        
    ## 2. 활동입력처리 부분 ##
    activity_list = []
    for act_i in act_txt_list:
        re = tokenized(act_i)
        activity_list.append(re)
    
    
    ## 전공관련 데이터셋 불러오기 ##
    data_major = pd.read_csv('total_major_fit_dataset.csv')
    
    ## 입력전공에 해당하는 세부항목 추출하기 ## >>>>>>>>>>> 세부항목을 비교할 거임
    extract_major = []
    for mj in major_list:
        if mj in data_major['major'].values: # 매칭되는 값이 있다면, 해당 행을 찾아서

            ext_index = data_major[data_major['major'] == mj] #조건에 맞는 값을 모두 가져온다.

            extract_major.append(ext_index)
    
    # extract_major 를 문자로 만들어서 개별 리스트로 만들자
    s =" ".join(map(str, extract_major))
    ext_input_major_query = s.split()# str을 공란()으로 구분하여 리스트로 만들고, 이것을 비교(전공의 카테로리, 활동내용의 카테고리)할거임
    ######====> ['Unnamed:',  '0',  'major',  'main_major',  '0',  '0',  'mechanical',  'engineering',......]
        
        
        
    ## 활동-관련 전공 데이터셋 불러오기 ##
    data_act = pd.read_csv('activity_major_db.csv')
    
    act_title = data_act['title']
    act_title_len = len(act_title)
    #print('act_title_len :', act_title_len)
    
    #입력데이터에서 추출한 활동내역(리스트)가 활동 타이틀에 있는지 확인한다.
    get_major = []
    n = 0
    if n <= len(act_title)-1: #모든 타이틀을 검사하기 위해서 총 개수와 같거나 작다면 아래 조건문 한번 실행  311개임
        for act_item in activity_list: # 입력한 활동관련 설명 10개의 리스트, 이것은 이미 사전에 토큰 처리됨(def activity_anaysis(text):)
            #print('act_item :', act_item)
            for j in act_item: #활동 관련 입력문장 개별 단어로 처리 잘됨
                #print('j :', j)          
                for m in list(data_act['title']): #타이틀의 n번째 문장을, 단어로 분해한 후 각 단어를 가져와서 활동관련 설명 단어와 비교한다.
                    #print('m :', m)
                    #토큰화한다. 그리고 다시  for 문으로 하나씩 그룹으로 꺼낸다.
                    tok = tokenized(m)
                    #print('tok :', tok)
                    for t in tok:         
                        if j == t: # 매칭되는게 있다면, 해당 전공관련 주제를 추출해본다.
                            print('FIT')
                            #print('j :', j) 
                            #print('n :', n)                  
                            major1 = data_act['big_major_category_1'][n]
                            get_major.append(major1)
                            get_major.append(n)
                            major2 = data_act['big_major_category_2'][n]
                            get_major.append(major2)
                            major3 = data_act['big_major_category_3'][n]
                            get_major.append(major3)
                            #print('N : ', n)
                        else:
                            pass
                            #print('NOT FIT')
                n += 1

    else:
        pass

    #get_major #추출한 전공이다. 이 전공 카테고리와 희망전공 3개입력한 내용의 전공분야별 카테고리를 비교해 볼 것이다.
    ext_mjr = set(get_major)
    print('활동으로부터 추출한 관련 전공 카테고리 : ', ext_mjr)
    ext_mjr_list = list(ext_mjr)
    
    rResult = []
    for z in ext_mjr_list: # 활동내역입력 내용에서 관련 전공 추출하여 하나씩 비교하여 
        for q in ext_input_major_query:#입력전공에 관련한 데이터베이스에서 관련 전공 카테고리 추출하여 하나씩 비교
            if z == q:
                rResult.append("FIT") # 같은것이 있으면 'FIT' 출력한다.
            else:
                rResult.append("TEST NOT FIT") ########################## 나중에 주석처리할 것임!!!!!!!!!!!!!!!!!!!!
                pass
    
    rResult = set(rResult)
    result_fin = list(rResult)
    
    return result_fin #최종결과 출력, 없으면 아무것도 출력하지 않음.  ================ 


############ 실행 함수  : (전공 3개(,로 구분), 활동내역 최대 10개까지) #####

def mjr_act_analy(input_text, input_activitys):

    result_fit = []
    for i in input_activitys:
        re = input_majors(input_text, i)
        result_fit.extend(re)

    # 총 활동 수
    act_numb = total_activity_num
    
    #각 활동의 전공 적합성 분석 결과 수
    major_fit = len(result_fit)

    # 점수 계산 로직
    if act_numb == 0:
        print ("weak: 1")
        result_fit = 1
    elif act_numb >= 1 and act_numb <= 2:
        if major_fit == 1:
            print ('average : 2')
            result_fit = 2
        elif major_fit == 0:
            print ('weak : 1')
            result_fit = 1
        else:
            print ('N/A')
    elif act_numb >= 3 and act_numb <=4:
        if major_fit >= 2:
            print ('superb : 5')
            result_fit = 5
        elif major_fit == 1:
            print ('average : 3')
            result_fit = 3
        else:
            print ('N/A')
    elif act_numb >= 6 and act_numb <= 7:
        if major_fit >= 3:
            print ('superb : 5')
            result_fit = 5
        elif major_fit == 2:
            print ('strong : 2')
            result_fit = 2
        elif major_fit == 1:
            print ('mediocre')
            result_fit = 1
        else:
            print ('N/A')
    elif act_numb >= 8 and act_numb <= 10:
        if major_fit >= 4:
            print ('superb: 5')
            result_fit = 5
        elif major_fit == 3:
            print ('strong : 4')
            result_fit = 4
        elif major_fit == 2:
            print ('average : 3')
            result_fit = 3
        elif major_fit == 1:
            print ('mediocre : 1')
            result_fit = 1
        else:
            print ('weak : 1')
            result_fit = 1
    else:
        pass



    return result_fit



############  실행 테스트 ##################################

re = mjr_act_analy(majors, total_actvity)


print("===================================================")
print('RESULT :', re)
print("===================================================")


# 결과


# ===================================================
# RESULT : 5
# ===================================================