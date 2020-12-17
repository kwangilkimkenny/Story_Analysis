###############  개별입력 EXTRACURRICULAR 계산 실행함수!!!~ 이것을 실행하면 됨, 그럼 나머지 함수들도 모두 연결되어 작동함

## 결과 : Dedication, Leadership, Description, Major Fit, Overall Strength 의 개별 결과값이 각각 산출됨

######################################################################################################################
# 실행함수  :   def total_desci_score(total_actvity, hrs_per_week, weeks_per_year, period_years_of_activity):

# 연결되어 있는 py 코드

# from ai_leadership import leadership_start_here, total_act_lists
# from ai_major_activity_fit  import mjr_act_analy, total_actvity, each_mjr_act_fit_analysis
# from ai_dedication_analysis  import dedication_analysis

############### 웹에서 입력해야 할 값 #################################################################################

##### 특별활동 시간 입력
# hrs_per_week = 5 
# weeks_per_year = 20 
# period_years_of_activity = 3 

##### 전공 입력 3개 ','로 구분하여 입력받아야 함, 대소문자 상관없음
# majors = """ mechanical engineering, Film Studies, Psychology  """

##### 활동내역 최대 1개가 입력되어야 함
# input_text_1 = """ 활동 내용입력 """

######################################################################################################################


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



from ai_leadership import leadership_start_here
from ai_description import total_desci_score

from ai_major_activity_fit  import mjr_act_analy, each_mjr_act_fit_analysis
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

input_text_2 = """"""

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
#############################################################################################################


def tot_input_act_number(inp_acti_list):
    total_activity_num = 0
    for a_item in inp_acti_list:
        if a_item:
            total_activity_num += 1
        else:
            pass
    return total_activity_num




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
        #print("개별항목 입력문자수:" , result_chr_count)
        result_each_char = round((result_chr_count / 36)*100,2)
        result_chr_count_.append(result_each_char)

        ##### 조건문으로 입력수에 대한 전체 평균값을 계산해야 함... description  강도 계산하기~!!!!!!!
    result_chr_count_ # 입력 항목별 개별점수 리스트  [94.44, 91.67, 52.78, 97.22, 66.67, 25.0, 55.56, 88.89, 0.0]

    #점수계산하기 (90%+ = 5점 / 75%-90% = 4점 / 60%-75% = 3점 / 40%-60% = 2점 / 40% 이하 = 1점)

    # 개별점수계산(항목별당 점수)
    each_item_score = []
    for k in result_chr_count_:
        if k >= 90:
            get_sc = '5'
            each_item_score.append(get_sc)

        elif k < 90 and k >=75:
            get_sc = '4'
            each_item_score.append(get_sc)

        elif k < 75 and k >= 60:
            get_sc = '3'
            each_item_score.append(get_sc)

        elif k < 60 and k >= 40:
            get_sc = '2'
            each_item_score.append(get_sc)

        else:
            get_sc = '1'
            each_item_score.append(get_sc)




    sum_re_chr_cnt = sum(result_chr_count_[:]) # 모든점수 합치기

    #입력하지 않은 값 제외
    only_input_value = 0
    for c_item in result_chr_count_:
        if c_item:
            only_input_value += 1
        else:
            pass
    

    print('디스크립션 계산결과 중간 리스트값 :' , result_chr_count_)

    #입력한 값에만 해당하는 것만으로 평균계산하기(입력하지 않은 값 제외하였음)
    char_mean = round(sum_re_chr_cnt /only_input_value, 2)

    print ('입력항목별 개별 점수 :', result_chr_count_) # [94.44, 91.67, 52.78, 97.22, 66.67, 25.0, 55.56, 88.89, 0.0]
    print ('입력항목(미입력 제외)의 전체 평균점수 :', char_mean) # 63.58
    print ('입력항목별 개별 최종 강도(점수) :', each_item_score) #  순서대로 [5, 5, 2, 5, 3, 1, 2, 4, 1]

    #점수가 높은 순서대로 인덱스값 추출 하기
    #상위 60% 이상의 입력값의 위치를 찾았음
    result_r = [element for array in each_item_score for element in array]
    #문자를 숫자로 변환
    re_r = list(map(float,result_r))
    desc_top6 = sorted(range(len(re_r)), key=lambda i: re_r[i], reverse=True)[:round(len(re_r)*0.6)]
    print ('입력항목별 상위 60%의 점수를 받은 항목들(높은 점수 순서대로 정렬) :', desc_top6)

    #([94.44, 91.67, 52.78, 97.22, 66.67, 25.0, 55.56, 88.89, 0.0], 63.58, ['5', '5', '2', '5', '3', '1', '2', '4', '1'], [0, 1, 3, 7, 4])
    return result_chr_count_, char_mean, each_item_score, desc_top6




####### 입력 활동 개별 점수 계산 실행 !
each_desc_score = character_counter_scoring(total_actvity)
print("==================================")
print ('입력활동 개별 점수 계산(순서대로) : ', each_desc_score)

###############   전체적인 overall_extracurricular 강도 계산 실행함수!!!~ 이것을 실행하면 됨, 그럼 나머지 함수들도 모두 연결되어 작동함
######################################################################################################################
def overall_extracurricular(majors, total_actvity, hrs_per_week, weeks_per_year, period_years_of_activity):
    tat_numb = tot_input_act_number(total_actvity) # 입력활동 수
    print('총 입력활동 수 : ', tat_numb)

    if tat_numb >=6 : # 입력한 내용이 6개 이상이라면
        print("5~10")

        leadership_re = leadership_start_here(total_actvity)
        leadership_re_value = leadership_re[0]
        print("LEADERSHIP : ", leadership_re_value)
        dedecation_re = dedication_analysis(hrs_per_week, weeks_per_year, period_years_of_activity)
        print("DEDICATION : ", dedecation_re)
        description_result = total_desci_score(majors, total_actvity, hrs_per_week, weeks_per_year, period_years_of_activity)
        print("DESCRIPTION :", description_result)
        re_each_M_A = each_mjr_act_fit_analysis(majors, total_actvity)
        print("MAJRO FIT score: ", re_each_M_A[1])
        check_fitness = re_each_M_A[-1] #결과물의 마지막 값이 fit, not fit 이다.
        print("전공적합성 FIT / NOT FIT :", check_fitness)         


         #MAJOR FIT 가산점 계산 부분
        fit_coef = re_each_M_A[1] 

        if fit_coef >= 5:
            add_point = 0.3
        elif fit_coef < 5 and fit_coef >= 4:
            add_point = 0.2
        elif fit_coef < 4 and fit_coef >= 3:
            add_point = 0.1
        elif fit_coef < 3 and fit_coef >= 2:
            add_point = -0.1
        else:
            add_point = -0.2

        #비율계산>>>>>>>>>>>>>>>>>>>>>>>>>>> 이게 최종 결과값임(조전문 하에서 말이징~)
        desc_strength_re = round(((leadership_re[0] * 0.6) + (dedecation_re * 0.4) + fit_coef)/3,2)
        print("Overall Strength : ", desc_strength_re)


    elif tat_numb <= 6 and tat_numb >= 4 :
        print("4~6")
        leadership_re = leadership_start_here(total_act_lists)
        leadership_re_value = leadership_re[0]
        print("LEADERSHIP : ", leadership_re_value)
        dedecation_re = dedication_analysis(hrs_per_week, weeks_per_year, period_years_of_activity)
        print("DEDICATION : ", dedecation_re)
        description_result = total_desci_score(majors, total_actvity, hrs_per_week, weeks_per_year, period_years_of_activity)
        print("DESCRIPTION :", description_result)
        re_each_M_A = each_mjr_act_fit_analysis(majors, total_actvity,tat_numb)
        print("MAJRO FIT : ", re_each_M_A[1])
        check_fitness = re_each_M_A[-1] #결과물의 마지막 값이 fit, not fit 이다.
        print("전공적합성 FIT / NOT FIT :", check_fitness)       

        #MAJOR FIT 가산점 계산 부분
        fit_coef = re_each_M_A[1] 

        if fit_coef >= 5:
            add_point = 0.3
        elif fit_coef < 5 and fit_coef >= 4:
            add_point = 0.2
        elif fit_coef < 4 and fit_coef >= 3:
            add_point = 0.1
        elif fit_coef < 3 and fit_coef >= 2:
            add_point = -0.1
        else:
            add_point = -0.2

        #비율계산>>>>>>>>>>>>>>>>>>>>>>>>>>> 이게 최종 결과값임(조전문 하에서 말이징~)
        desc_strength_re = round(((leadership_re[0] * 0.6) + (dedecation_re * 0.4) + fit_coef)/3,2)
        print("Overall Strength : ", desc_strength_re)

    elif tat_numb <= 3 and tat_numb >= 1 :
        print("1~4")
        leadership_re = leadership_start_here(total_actvity)
        leadership_re_value = leadership_re[0]
        print("LEADERSHIP : ", leadership_re_value)
        dedecation_re = dedication_analysis(hrs_per_week, weeks_per_year, period_years_of_activity)
        print("DEDICATION : ", dedecation_re)
        description_result = total_desci_score(majors, total_actvity, hrs_per_week, weeks_per_year, period_years_of_activity)
        print("DESCRIPTION :", description_result)
        re_each_M_A = each_mjr_act_fit_analysis(majors, total_actvity)
        print("MAJOF FIT : ", re_each_M_A[1])
        check_fitness = re_each_M_A[-1] #결과물의 마지막 값이 fit, not fit 이다.
        print("전공적합성 FIT / NOT FIT :", check_fitness)       

        #MAJOR FIT 가산점 계산 부분
        fit_coef = re_each_M_A[1] 

        if fit_coef >= 5:
            add_point = 0.3
        elif fit_coef < 5 and fit_coef >= 4:
            add_point = 0.2
        elif fit_coef < 4 and fit_coef >= 3:
            add_point = 0.1
        elif fit_coef < 3 and fit_coef >= 2:
            add_point = -0.1
        else:
            add_point = -0.2
        

        # [Overall Dedication level (60%) + Overall Leadership level (30%) + Overall Description Level (10%)] + Major Fit 가산점 
        # 비율계산>>>>>>>>>>>>>>>>>>>>>>>>>>> 이게 최종 결과값임(조건문 하에서 말이징~)
        desc_strength_re = round(((leadership_re[0] * 0.6) + (dedecation_re * 0.3) + fit_coef)/3 + add_point ,2)
        print("Overall Strength : ", desc_strength_re)

    else:
        print("0")
        pass


    final_result = [dedecation_re, leadership_re_value, description_result, fit_coef, check_fitness, desc_strength_re]

    # 최종값

    return final_result


# ==== 이 부분 개발적용 해야 함 ====

# -가장 Leadership+Dedication+Major Fit 점수 합계가 높은 상위 60% 활동만 계산에 활용 (각 활동별로 Dedication level (60%) + Leadership level (40%) + Major Fit
# 가산점 더해서 가장 높은 순위부터 계산에 활용… 중요한 활동부터)
# (10개 입력시 = 상위 6개 활동 활용, 8개 = 5개 (반올림), 6개 = 4개)

# -총 200 캐릭터 중 채운 캐릭터의 수의 고려 (고려하는 모든 활동의 캐릭터 수를 더해서 평균을 내도 되구요, 아니면 각각 점수를 구해서 평균을 내셔도 되구요… 
# 단어 사이의 space는 빈칸이 아님. 마지막 단어 이후의 space는 빈칸임, 글자가 없으므로)
# (90%+ = 5점 / 75%-90% = 4점 / 60%-75% = 3점 / 40%-60% = 2점 / 40% 이하 = 1점)





####### 입력 활동 개별 점수 계산 실행 !
# tot_desc_score = total_desci_score(total_actvity, hrs_per_week, weeks_per_year, period_years_of_activity)
# print("==================================")
# print ('입력활동 전체적인 점수 계산: ', tot_desc_score)

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





# ===========================================================================================================
##########################     Overall Extracurricular 강도 계산 실행 테트스   ###############################

result_overall  = overall_extracurricular(majors, total_actvity, hrs_per_week, weeks_per_year, period_years_of_activity)

print ("=================================")
print ("RESULT overall_extracurricular:", result_overall)
print ("=================================")

# ===========================================================================================================

# 실행결과 : 순서대로 Dedication, Leadership, Description, Major Fit score, Major Fit, Overall Strength

# =================================
# RESULT overall_extracurricular: [3.38, 4.0, 1.58, 1.0, ['NOT SURE'], 1.27]
# =================================