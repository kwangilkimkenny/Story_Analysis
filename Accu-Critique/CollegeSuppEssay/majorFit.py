# major fit
# 1) get major words from essay
# 2) selected prompt



import re
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import multiprocessing
from gensim.models import Phrases
import gensim

#데이터 전처리 
def cleaning(essay_input):
    #입력한 글을 모두 단어로 쪼개로 리스트로 만들기 - 
    essay_input_corpus_ = str(essay_input) #문장입력
    essay_input_corpus_ = essay_input_corpus_.lower()#소문자 변환

    sentences_  = sent_tokenize(essay_input_corpus_) #문장단위로 토큰화(구분)되어 리스에 담김

    # 문장을 토크큰화하여 해당 문장에 Verbs가 있는지 분석 부분 코드임 

    split_sentences_ = []
    for sentence in sentences_:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences_.append(words)
        
    # 입력한 문장을 모두 리스트로 변환
    input_text_list = [y for x in split_sentences_ for y in x] # 이중 리스트 Flatten
    result = list(set(input_text_list))
    result_include_double = input_text_list # 중복이 저게되지 않은 리스트
    return result, essay_input_corpus_, result_include_double


def majorfit(essay_input):

    cln_re = cleaning(essay_input)
    cln_essay = cln_re[0]
    cnl_essay_all = cln_re[2] # 중복제거되지 않은, 토큰화된 에세이
    #print('cln_essay:', cln_essay)
    cnl_sents_ = cln_re[1] # 토큰화하지 않은 문자열(에세이 전체)

    # load summer activities data
    summer_activities = pd.read_csv("./data/SummerPrograms.csv")
    #소문자로 변환
    summer_activities['title'] = summer_activities['title'].str.lower() 
    summer_activities['1st_Major_Category'] = summer_activities['1st_Major_Category'].str.lower()
    summer_activities['2nd_Major_Category'] = summer_activities['2nd_Major_Category'].str.lower()
    summer_activities['3nd_Major_Category'] = summer_activities['3nd_Major_Category'].str.lower()
    #    class	title                                   score	1st_Major_Category	2nd_Major_Category	3nd_Major_Category
    # 0	SUMMER	rsi (research science institute) at mit	5	    math/science	    tech/engineering	NaN
    # 1	SUMMER	mit women's technology program (wtp)	5	    math/science	    tech/engineering
    #  ...

    #title을 인덱스로 변환, 그래야 값을 찾기 쉽다.
    summer_activities.set_index('title', inplace=True)
    # title	                 score_cal_rate	  fin_score	
    # extremely selective	5	              5.0
    # very selective	    4	              4.0
    # ...
    #print(summer_activities)


    # 단어리스트 최기화 설정
    get_score__ = []
    get_score___ = []
    get_word_position = [] # 이것이 중요함! 추출한 단어들의 응집성(위치가 조밀하면 해당 활동일 가능성이 높음)
    for i in cln_essay: # 에세이에서 단어를 하나씩 가져와서
        cnt = 0  # 카운토 초기화
        for j in summer_activities.index: #인덱스의 summer activity major를 하나씩 꺼내와서
            sum_act_wd = j.split() # 단어로 분리
            #print('len__sum_act_wd:', len(sum_act_wd))

            if i in sum_act_wd: # 활동명의 각 단어를 꺼내와서
                cnt += 1
                #essay를 리스로 변환후 포함 여부 비교, 있다면 해당 점수를 가져온다.
                get_wd_posi = cnl_sents_.find(i) # 활동명의 개별 단어가, 에세이에서 어디에 위치해 있는지 파악한다.
                get_word_position.append(get_wd_posi)# 위치값 추출한다.
                if cnt <=  len(sum_act_wd): # 인덱스의 활동 명의 단어 수보다 -4 적은 수가 일치한다면(활동명칭에서 summer, program을 제거했기때문에 숫자 -4를 적음)
                    #print('cnt :', cnt)
                    # 문자열 완전일치 판단
                    if j in cnl_sents_: # 에세이에서 활동정보(j)과 일치하는 문자열이 있는지 확인, 있다면 이하 수행 - 활동명, 스코어 추출하여 리턴
                        get_score = summer_activities.loc[j, 'score']
                        get_score__.append(j)
                        get_score___.append(get_score)
                else:
                    pass

    # 계산한 결과와 에세이 본문의 문장들과의 일치율 계산하기
    gwp_re = list(set(get_word_position)) # 중복제거
    # detect_activities = gwp_re # 활동명의 단어들이 입력에세이의 어떤 위치값을 가지는지 확인

    # 추출한 값의 중복값 제거
    get_score_fin = list(set(get_score__)) # 추출한 summer activity 명칭
    # print('get_score_fin:', get_score_fin) # 에세이서 발견한 활동명칭
    get_score_fin_re = list(set(get_score___))
    # print('get_score_fin_re :', get_score_fin_re) # [5] 로 결과가 리스트 값으로 나오기때문에 [0]번째의 데이터를 꺼내서 비교
    get_score_fin_re = get_score_fin_re[0]

    # Major fit 
    mjr_1st = summer_activities.loc[get_score_fin, '1st_Major_Category'].to_string() # 활동명(get_score_fin)에 해당하는 전공명 추출
    mjr_1st = mjr_1st.split()
    mjr_1st = mjr_1st[-1:]
    # print('mjr_1st :', mjr_1st)
    mjr_2nd = summer_activities.loc[get_score_fin, '2nd_Major_Category'].to_string() # 활동명(get_score_fin)에 해당하는 전공명 추출
    mjr_2nd = mjr_2nd.split()
    mjr_2nd = [mjr_2nd[-1]]
    mjr_3nd = summer_activities.loc[get_score_fin, '3nd_Major_Category'].to_string() # 활동명(get_score_fin)에 해당하는 전공명 추출
    mjr_3nd = mjr_3nd.split()
    mjr_3nd = [mjr_3nd[-1]]

    extracted_major = [mjr_1st, mjr_2nd, mjr_3nd] # 추출한 전공들
    extracted_major = [y for x in extracted_major for y in x]
    extracted_major = " ".join(extracted_major)
    extracted_major = extracted_major.replace("/",  " ")
    extracted_major = word_tokenize(extracted_major)
    # print('extracted_major:', extracted_major) # extracted_major: ['math', 'science', 'tech', 'engineering', 'NaN']

    # 에세이이 중복 단어가 몇개가 되는지 - 전공관련 단어가 몇개있는지 파악하기 위함
    w_cnt = {}
    for lst in cnl_essay_all:
        try: w_cnt[lst] += 1
        except: w_cnt[lst] = 1
    # print('에세이의 중복단어 수를 딕셔너리로 표현:', w_cnt)
    #에세이의 중복단어 수를 딕셔너리로 표현: {'i': 44, 'inhale': 1, 'deeply': 2, 'and': 27, 'blow': 1 ...


    matched_major = [] # 에세이와 매칭되는 섬머활동 관련 전공
    extracted_mjr_all_nums = [] # 추출된 점수
    for mjr_itm in extracted_major: # 추출된 전공을 하나씩 꺼내서
        if mjr_itm in cnl_essay_all: # 에세이에 전공명칭이 있다면
            matched_major.append(mjr_itm) # 에세이에 포함된 전공을 모은다. 그리고 이 전공이 에세이이 얼마나 빈번하게 등장하는지 세어본다.
            for dic_itm in w_cnt: # 딕셔너리의 키값(단어)을 하나씩 꺼내와서 전공관련 단어가 몇개가 있는지 확인
                # print('dic_itm', dic_itm)
                if dic_itm == mjr_itm: # 키값(단어)이 전공명과 같다면
                    extract_mjr_num = w_cnt[dic_itm] # value값을 추출한다
                    extracted_mjr_all_nums.append(extract_mjr_num)
                else:
                    extract_mjr_num = 0
                    extracted_mjr_all_nums.append(extract_mjr_num)
        else:
            extract_mjr_num = 0
            extracted_mjr_all_nums.append(extract_mjr_num)

    rre = sorted(extracted_mjr_all_nums, reverse=True)
    extract_mjr_num = rre[0] # 가장 큰 수를 추출
    # print('extract_mjr_num:', extract_mjr_num)

    # 추출한 점수를 5가지 척도로 변환하기.
    if extract_mjr_num  == 5:
        result_sc = 'Supurb'
        result_score = 100
    elif extract_mjr_num == 4:
        result_sc = 'Strong'
        result_score = 80
    elif extract_mjr_num == 3:
        result_sc = 'Good'
        result_score = 60
    elif extract_mjr_num == 2:
        result_sc = 'Mediocre'
        result_score = 40
    else:
        result_sc = 'Lacking'
        result_score = 20

    data = {
        'Major Score' : result_sc, # 전공접합성의 매칭되는 결과로 점수로 정하기 5점 척도로 표현
    }

    return data
