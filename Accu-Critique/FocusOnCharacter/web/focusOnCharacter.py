# 아나콘다 가상환경 office:  py37TF2
# home : py37Keras

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
import spacy

import itertools
    
nlp = spacy.load('en_core_web_lg')

# 이름으로 인물 수 계산
def find_named_persons(text):
    # Create Doc object
    doc2 = nlp(text)
    # Identify the persons
    persons = [ent.text for ent in doc2.ents if ent.label_ == 'PERSON']
    #총 인물 수
    get_person = len(persons)
    # Return persons
    return get_person


def characters(input_text):
    #소문자로 변환
    input_lower_text = input_text.lower()
    about_doc = nlp(input_lower_text)
    token_list = {}
    for token in about_doc:
        #print (token, token.idx)
        token_list.setdefault(token, token.idx)
    
    li_doc = list(token_list.keys())

    #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고
    i_character_list = ['i', 'my', 'me', 'mine']
    
    #하나씩 꺼내서 유사한 단어를 찾아내서 새로운 리스트에 담아서 출력,
    ext_i_characters = []
    for i_itm in i_character_list:
        for k_ in li_doc:
            if i_itm == str(k_):
                ext_i_characters.append(i_itm)
                
    #I 관련 캐릭터 표현하는 단어들의 총 개수        
    get_i = len(ext_i_characters)

    #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고,
    you_character_list = ['you', 'your', 'they','them',
                    'yours', 'he','him','his' 'she','her','it','someone','their', 'myself', 'aunt',
                    'brother','cousin','daughter','father','grandchild','granddaughter','granddson','grandfather',
                    'grandmother', 'person','great-grandchild','husband','ex-husband','son-in-law', 'daughter-in-law','mother',
                    'niece','nephew','parents','sister','son','stepfather','stepmother','stepdaughter', 'stepson',
                    'twin','uncle','widow','widower','wife','ex-wife']
    
    #하나씩 꺼내서 유사한 단어를 찾아내서 새로운 리스트에 담아서 출력,
    ext_you_characters = []
    for i_itm in you_character_list:
        for k_ in li_doc:
            if i_itm == str(k_):
                ext_you_characters.append(i_itm)
                
    #I 관련 캐릭터 표현하는 단어들의 총 개수        
    get_others = len(ext_you_characters)

    return get_i, get_others


def character_words(input_text):
    #소문자로 변환
    input_lower_text = input_text.lower()
    about_doc = nlp(input_lower_text)
    token_list = {}
    for token in about_doc:
        #print (token, token.idx)
        token_list.setdefault(token, token.idx)
    
    li_doc = list(token_list.keys())

    #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고
    i_character_list = ['i', 'my', 'me', 'mine']
    
    #하나씩 꺼내서 유사한 단어를 찾아내서 새로운 리스트에 담아서 출력,
    ext_i_characters = []
    for i_itm in i_character_list:
        for k_ in li_doc:
            if i_itm == str(k_):
                ext_i_characters.append(i_itm)
                
    #I 관련 캐릭터 표현하는 단어들        
    get_i_words = ext_i_characters

    #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고,
    you_character_list = ['you', 'your', 'they','them',
                    'yours', 'he','him','his' 'she','her','it','someone','their', 'myself', 'aunt',
                    'brother','cousin','daughter','father','grandchild','granddaughter','granddson','grandfather',
                    'grandmother', 'person','great-grandchild','husband','ex-husband','son-in-law', 'daughter-in-law','mother',
                    'niece','nephew','parents','sister','son','stepfather','stepmother','stepdaughter', 'stepson',
                    'twin','uncle','widow','widower','wife','ex-wife']
    
    #하나씩 꺼내서 유사한 단어를 찾아내서 새로운 리스트에 담아서 출력,
    ext_you_characters = []
    for i_itm in you_character_list:
        for k_ in li_doc:
            if i_itm == str(k_):
                ext_you_characters.append(i_itm)
                
    #I 관련 캐릭터 표현하는 단어들의 총 개수        
    get_others_words = ext_you_characters

    return get_i_words, get_others_words


#########################################################
# 650단어에서 또는 전체 단어에서 단락별 셋팅단어 활용 수 분석
# 20% intro, 60% body1,2,3 20% conclusion
##########################################################
    
def paragraph_divide_ratio(text):

    essay_input_corpus = str(text) #문장입력
    essay_input_corpus = essay_input_corpus.lower()#소문자 변환

    sentences  = word_tokenize(essay_input_corpus) #문장 토큰화
    # print('sentences:',sentences)

    # 총 문장수 계산
    total_sentences = len(sentences) # 토큰으로 처리된 총 문장 수
    total_sentences = float(total_sentences)
    #print('total_sentences:', total_sentences)

    # 비율계산 시작
    intro_n = round(total_sentences*0.2) # 20% 만 계산하기, 소수점이하는 반올림
    body_1 = round(total_sentences*0.2) # 20% 만 계산하기, 소수점이하는 반올림
    body_2 = round(total_sentences*0.2)
    body_3 = round(total_sentences*0.2)
    conclusion_n = round(total_sentences*0.2) # 20% 만 계산하기, 소수점이하는 반올림

    #데이터셋 비율분할 완료
    intro = sentences[:intro_n]
    # print('intro :', intro)
    body_1_ = sentences[intro_n:intro_n + body_1]
    # print('body 1 :', body_1_)
    body_2_ = sentences[intro_n + body_1:intro_n + body_1 + body_2]
    #print('body 2 :', body_2_)
    body_3_ = sentences[intro_n + body_1 + body_2:intro_n + body_1 + body_2 + body_3]
    # print('body_3_ :', body_3_)
    conclusion = sentences[intro_n + body_1 + body_2 + body_3 + 1 :]
    # print('conclusion :', conclusion)
    
    #print('sentences:', sentences)
    #데이터프레임으로 변환
    df_sentences = pd.DataFrame(sentences,columns=['words'])
    #print('sentences:',df_sentences)
    
    
    
    
    # 캐릭터 관련 단어 추출하기
    doc2 = nlp(text)
    # Identify the persons
    persons = [ent.text for ent in doc2.ents if ent.label_ == 'PERSON']
    #print('person_name: ', persons)
    
    #이름 소문자 변환
    lower_all_names = []
    for p in persons:
        lower_names = p.lower()
        lower_all_names.append(lower_names)
    
    # 추출한 단어의 위치를 찾아내기
    name_position = []
    for name in persons: # 추출한 값을 하나씩 꺼내서 전체 문장에서의 위치를 찾는다.
        name = name.lower()
        n_position = df_sentences.loc[(df_sentences['words'] ==  name)]
        name_position.append(n_position)     
    #print('name_position :', name_position)
    name_all_position_index = [] # 이름의 위치를 리스트에 담아보자
    for nm_df in name_position: # 아이템을 리스트에서 불러와서
        name_all_position_index.append(nm_df.index) # 데이테프레임의 인덱스값을 추출하여 리스트에 담는다.
    #print('name_all_position_index', name_all_position_index)
        
    #이제 리스트의 값을 하나씩 가져와서 숫자만 추출하자
    name_words_all_position =[]
    for numpy_nm in name_all_position_index: #numpy이기 때문에 np.append를 사용하여 값을 추출하여 append한다.
        name_words_all_position = (np.append(name_words_all_position, numpy_nm, axis=0)).tolist()
        
    ###### 1)문장내 이름의 위치값을 리스트로 추출 ######
    #print('names 단어의 위치값:', name_words_all_position)
        
        
   

    charater_words = list(character_words(input_text))
    charater_words = sum(charater_words, [])
    charater_words = set(charater_words) #중복제거
    charater_words = list(charater_words)
    #print('charater_words: ', charater_words)
    #print('len:', len(charater_words))


    #character_words의 각 단어로 데이터프레임의 컬럽명으로 인덱스 찾아서 리스트에 담기
    ch_position = []
    for cht_word in charater_words:
        req_index_position = df_sentences.loc[(df_sentences['words'] ==  cht_word)]
        ch_position.append(req_index_position)
    #print('ch_position:', ch_position) # 여기서 인덱스 숫자만 추출하기, 이 숫자가 캐릭터단어의 리스트
    #print('dtype:', type(ch_position)) # 리스트안에 데이터프레임이 들어있다. 이것을 꺼내려면 리스트를 불러와서 데이터프레임을 조작해야함
    words_position_li = []
    for li_df  in ch_position: # 아이템을 리스트에서 불러와서
        words_position_li.append(li_df.index) #데이테프레임의 인덱스값을 추출하여 리스트에 담는다.
        
    #print("words_position_li:", words_position_li)#잘됨
    #이제 리스트의 값을 하나씩 가져와서 숫자만 추출하자
    
    character_words_all_position = [] #캐릭터를 의미하는 단어들의 위치를 모두 추출하여 정리한 결과!!  저장성공!!!
    for numpy_itm in words_position_li:
        #numpy이기 때문에 np.append를 사용하여 값을 추출하여 append한다.
        character_words_all_position = (np.append(character_words_all_position, numpy_itm, axis=0)).tolist()
   
    ###### 2)문장내 캐릭터의 위치값을 리스트로 추출 ######        
    #print('캐릭터를 표현한 단어들의 위치값:', character_words_all_position)
        
    ###### 3) '이름 + 캐릭터 표현 단어'를 모두 합친 최종 리스트 
    name_and_character_all_list = name_words_all_position + character_words_all_position
    #print('이름 + 캐릭터 표현 단어를 모두 합친 최종 리스트 :', name_and_character_all_list)
    name_and_character_all_list = list(map(int, name_and_character_all_list)) # 정수를 실수로 변환
    #print('이름 + 캐릭터 표현 단어를 모두 합친 최종 리스트 :', name_and_character_all_list)
    #print('type:', type(name_and_character_all_list))
    # 오름차순 순서대로 정렬
    name_and_character_all_list = sorted(name_and_character_all_list, reverse=False)
    #print('이름 + 캐릭터 표현 단어를 모두 합친 최종 리스트(오름차순정렬) :', name_and_character_all_list)
    # character 관련 단어가 들어간 문장을 추출한다.(이건 html 밑줄용)
    # 1) 캐릭터 관련 모든 단어 추출하기, 추출한 단어의 위치 찾기(리스트의 번호)
    # 2) 리스트의 단어를 문장단위로 구분하기, 문장의 위치 찾기(html 밑줄용)
    # 3) 구간별 단어를 모두 추출하여 웹페이지에 표시할 것(언더라인 데코레이션으로 단락 구분표시해야 함)
    
    # name_and_character_all_list 이것이 전체 문장에서 어디에 위치해 있는지 인덱스값을 구간별(intro, body1~3, conclusion)로 나눈다.
    
    # 캐릭터표현 + 이름 리스트 
    totla_char_person_li = charater_words + lower_all_names
    #print('total 캐릭터표현 + 이름 리스트:', totla_char_person_li)
    
    # 추출한 리스트를 가지고 전체 문장에서 추출한 리스트의 인덱스(위치값)을 추출하기
    # char_name_re_data : 이것은 캐릭터, 이름의 위치값으로 벨류(단어만 추출해야 함)
    # inputData : 구간값(intro, body1~3, conclusion)으로 리스트임
    def counter_char_expression_in_parts(char_name_re_data, inputData):
        #print('캐릭터 설명 단어들:', char_name_re_data)
        # 겹치는 값이 있는지 확인(구간별 캐릭텨 표현 단어가 있는지 확인하고, 있다면 몇개가 존재하는지 계산)
        result_included_words = [] # 해당 구간에 포함된 캐릭터 표현단어 리스트
        for s1 in char_name_re_data:
            if s1 in inputData:
                result_included_words.append(s1)
                
        count_ch_words = len(result_included_words)
        return count_ch_words
    
    get_result_of_charter_expression_of_part = []
    save_vaue_and_positon =[]
    ############# 정확히 추출되는지 확인할 것 ###############################
    intro_position = counter_char_expression_in_parts(totla_char_person_li, intro)
    # print('intro Position:', intro_position)
    get_result_of_charter_expression_of_part.append(intro_position)
    save_vaue_and_positon.append(intro_position)
    save_vaue_and_positon.append('intro')
    
    body_1_position = counter_char_expression_in_parts(totla_char_person_li, body_1_)
    print('body1 Position:', body_1_position)
    get_result_of_charter_expression_of_part.append(body_1_position)
    save_vaue_and_positon.append(body_1_position)
    save_vaue_and_positon.append('body #1')
    
    body_2_position = counter_char_expression_in_parts(totla_char_person_li, body_2_)
    # print('body2 Position:', body_2_position)
    get_result_of_charter_expression_of_part.append(body_2_position)
    save_vaue_and_positon.append(body_2_position)
    save_vaue_and_positon.append('body #2')
    
    body_3_position = counter_char_expression_in_parts(totla_char_person_li, body_3_)
    # print('body3 Position:', body_3_position)
    get_result_of_charter_expression_of_part.append(body_3_position)
    save_vaue_and_positon.append(body_3_position)
    save_vaue_and_positon.append('body #3')
    
    conclusion_position = counter_char_expression_in_parts(totla_char_person_li, conclusion)
    # print('conclusion Position:', conclusion_position)
    get_result_of_charter_expression_of_part.append(conclusion_position)
    save_vaue_and_positon.append(conclusion_position)
    save_vaue_and_positon.append('conclusion')
    
    result_ = sorted(get_result_of_charter_expression_of_part, reverse=True)
    
    #각 구간별 결과값 중 가장 큰 결과값이 나온 구간을 추출 비교한다.
    print('save_vaue_and_positon:', save_vaue_and_positon)
    #result_[0] 가장 큰 값으로 가장 많이 사용한 구간에서의 단어 활용 총 합
    
    
    getParts = result_[0]
    getParts_2nd = result_[1]
    
    get_index_part = save_vaue_and_positon.index(getParts)
    get_index_part_2nd = save_vaue_and_positon.index(getParts_2nd)
    
    f_result = save_vaue_and_positon[int(get_index_part)+1]
    f_result_2n = save_vaue_and_positon[int(get_index_part_2nd)+1]
    # print(f_result)
    # name_and_character_all_list : 이름과 캐릭터표현 단어가 위치한 곳은 인덱값
    # return name_and_character_all_list, intro, body_1, body_2, body_3, conclusion
    
    
    ############################
    #### 구간들이 합격평균과 일치 ####
    ############################
    print('개인에세이 단락 캐릭터 점수:', result_)
    
    #### 그룹에세이 단락 캐릭터 포함 점수(임의로 넣음) ####
    group_parts_mean = [3,4,7,8,8]
    print('그룹에세이 단락 캐릭터 점수:', group_parts_mean)
    
    
    def simility(a, b):
        re_c = abs(a - b)
        if re_c  == 0: # 구간이 일치하면 
            fit = 'fit'
        elif re_c <= 5 and re_c > 2:
            fit = 'fit'
        elif re_c <= 10 and re_c > 6:
            fit = 'not fit'
        else: # 차이가 크면
            fit = 'not fit'
            
    intro_sim_re = simility(result_[0], group_parts_mean[0])
    body_1_sim_re = simility(result_[1], group_parts_mean[1])
    body_2_sim_re = simility(result_[2], group_parts_mean[2])
    body_3_sim_re = simility(result_[3], group_parts_mean[3])
    conclusion_sim_re = simility(result_[4], group_parts_mean[4])
    
    if intro_sim_re == 'fit' and body_1_sim_re == 'fit' and body_2_sim_re == 'fit' and body_3_sim_re == 'fit' and conclusion_sim_re == 'fit': 
        sentence_2nd_emp_sec = ['Comparing this with your essay, we see a very similar pattern.']
    elif intro_sim_re == 'fit' or body_1_sim_re == 'fit' or body_2_sim_re == 'fit' or body_3_sim_re == 'fit' or conclusion_sim_re == 'fit': 
        sentence_2nd_emp_sec = ['Comparing this with your essay, we see some similarities in the pattern.']
    else:
        sentence_2nd_emp_sec = ['Comparing this with your essay, we see a different pattern.']
    
    #### 결과해석 ####
    # result_ : 구간별 개인 에세이 캐릭터 등장 비율 ex)[3,4,7,8,8]
    # group_parts_mean : 구간별 그룹 평균 에세이 캐릭터 비율
    # sentence_2nd_emp_sec : Emphasis Character by Section (# of Character Descriptors)의 두번째 문장
    # f_result : 가장 캐릭터가 많이 등장한 구간
    # f_result_2n : 두번째로 캐릭터가 많이 등장한 구간
    
    return f_result, f_result_2n, result_, group_parts_mean, sentence_2nd_emp_sec


# input_text : 입력에세이
# promt_no : 선택질문  >> ex) 'prompt_1'...
# intended_character : mostly me >> 'me' = 1, me & some others : 'meAndOtehrs' = 2, other characters: 'others' = 3
# intended_character의 입력은 'me', 'meAndOtehrs', 'others'

def focusOnCharacters(input_text, promt_no, intended_character):

    person_num = find_named_persons(input_text)
    charater_num = list(characters(input_text))
    print('character_num:', charater_num)
    
    # ppt >> Number of Charactr Descriptors(graph로 표현해야 하는 값)
    total_character_descriptors_personal = sum(charater_num) # 개인 에세이에서 분석 추출한 총 캐릭터 표현 수
    descriptors_about_yourself = charater_num[0] #개인 에세이 추출 표현 about i
    
    total_character_descriptors_group = 40 ####### 1000명의 에세이에서 공통적으로 추출계산한 캐릭터 총 평균값(임의로 정함, 계산후 넣어야 함)
    descriptors_about_others_group = 10 ###### 1000명의 에세이 추출 others 캐릭터 평균값(임의로 정했음, 계산후 넣어야 함)
    
    
    # ppt >> Emphasis on You vs. Others
    admitted_case_avg = [35, 65] # you(I) : 35%, others: 65% >>>>>>>> 합격한 학생들의 평균갑으로 임의로 넣음(나중에 계산해서 넣어야 함)
    your_essay_you_vs_others = charater_num
    
    
    # I에 관련한 단여 사용 평균 평가결과분석(개인/그룹)
    # your_essay_you_vs_others[0] : 개인you사용값
    # admitted_case_avg[0] : group 평균값
    if your_essay_you_vs_others[0] > admitted_case_avg[0]:
        words_num = abs(your_essay_you_vs_others[0] - admitted_case_avg[0]) #평균값과 단어활용 수 차이 계산
        words_num_few_more = [words_num, "more"]
    elif abs(your_essay_you_vs_others[0] - admitted_case_avg[0]) is 0: #차이가 없다면
        words_num_few_more = ['fit']
    else:
        words_num = abs(your_essay_you_vs_others[0] - admitted_case_avg[0]) #평균값과 단어활용 수 차이 계산
        words_num_few_more = [words_num, "fewer"]
    
    # i, others 관련 단어 사용 비율 계산하여 3nd  >> emp_sentence3
    self_description_percent = [round((your_essay_you_vs_others[0] / your_essay_you_vs_others[1]) * 100)]
    
    other_description_percent = [round((your_essay_you_vs_others[1] / your_essay_you_vs_others[0]) * 100)]
    
    
    # input prompt number. 결과물 seleected_prompt_number을 sentece4에 적용
    selected_prompt_number = []
    if promt_no == "promt_1":
        selected_prompt_number.append("prompt #.1")
    elif promt_no == "promt_2":
        selected_prompt_number.append("prompt #.2")
    elif promt_no == "promt_3":
        selected_prompt_number.append("prompt #.3")
    elif promt_no == "promt_4":
        selected_prompt_number.append("prompt #.4")
    elif promt_no == "promt_5":
        selected_prompt_number.append("prompt #.5")
    elif promt_no == "promt_6":
        selected_prompt_number.append("prompt #.6")
    elif promt_no == "promt_7":
        selected_prompt_number.append("prompt #.7")
    else:
        pass
    
    # print('selected prompt number:', selected_prompt_number)
    
    #intended character define
    if intended_character == 'me': # me
        intended_character_ = 1
    elif intended_character == 'meAndOthers': # me and others
        intended_character_  = 2
    else: # others
        intended_character_  = 3

    sum_character_num = person_num + charater_num[0] + charater_num[1]
    ratio_i = round((charater_num[0] / sum_character_num),2) * 100

    if ratio_i >= 70: # i 가 70% 이상
        # print("Mostly Me")
        result = 1 # "Mostly Me"
    elif 40 <= ratio_i < 70: # i가 40~ 70% 
        # print("Me & some others")
        result = 2 # "Me & some others"
    else:
        # print("Others characters") # i가 40% 이하
        result = 3 # "Others characters"
      
    
    # 첫 문장 생성하기    
    # Focus on Character(s) by Admitted Students for
    admitted_student_for = selected_prompt_number
    
    # 1, 2nd Senctece 생성
    if result == 1:
        sentence1 = ['Regarding the number of characters, you’ve intended to focus on yourself mainly.']
        sentence2 = ['The AI analysis indicates that your personal statement seems to be focusing mostly on you.']
    
    elif result == 2:
        sentence1 = ['Regarding the number of characters, you’ve intended to focus mainly on a small number of characters, including yourself.']
        sentence2 = ['The AI analysis indicates that your personal statement seems to be focusing mostly on you and some other characters.']
    
    elif result == 3:
        sentence1 = ['Regarding the number of characters, you’ve intended to focus mainly on the people around you.']
        sentence2 = ['The AI analysis indicates that your personal statement seems to be focusing mostly on other characters.']
    else:
        pass
    
    # 3nd sentence
    if intended_character_ == result: #1&2가 같을 경우
        sentence3 =["Overall, the number of characters and description in your essay seems to be coherent with what you have intended."]
    elif intended_character_  == 1: # mostly me
        sentence3 = ['If you wish to shift the essay’s direction towards your original intention, you may consider reducing the number of characters and including more intrapersonal aspects.']
    elif intended_character_ == 2 and result == 1: # Me and some others (#2가 Mostly Me 인 경우)
        sentence3 = ['If you wish to shift the essay’s direction towards your original intention, you may consider increasing the number of characters and including more interactions between a few core members.']
    elif intended_character_ == 2 and result == 3: # Me and some others (#2가 Other characters 인 경우)
        sentence3 - ['If you wish to shift the essay’s direction towards your original intention, you may consider reducing the number of characters. You may focus more on your own thoughts and interaction between a few core characters in the story.']
    elif intended_character_ == 3: #Other characters
        sentence3 = ['If you wish to shift the essay’s direction towards your original intention, you may consider increasing the number of characters and including more interactions between members.']
    else:
        pass
    
    #합격케이스의 평균값 입력
    adm_mean_result = 2 # 합격케이스 평균값 설정(이것은 1000명의 통계를 돌려서 결과를 반영해야함. 지금은 임의값 적용하였음)
    
    #비교항목
    admit_mean_value_mostly_me = 1
    admit_mean_value_me_and_a_few_others = 2
    admit_mean_value_multiple_characters = 3
    
    # 4nd sentence  합격케이스 평균값 적용
    if adm_mean_result == admit_mean_value_mostly_me:
        sentence4 = ['The admitted case trend indicates many applicants tend to focus on themselves for', selected_prompt_number]
    elif adm_mean_result == admit_mean_value_me_and_a_few_others:
        sentence4 = ['The admitted case trend indicates many applicants tend to focus on themselves and a few other core characters for', selected_prompt_number]
    elif adm_mean_result == admit_mean_value_multiple_characters:
        sentence4 = ['The admitted case trend indicates many applicants tend to focus on other characters for', selected_prompt_number]
    else:
        pass
    
    #5th sentence
    #Intended = 합격케이스랑 match
    if intended_character_ == result: #Intended = 합격케이스랑 match 그리고 detected = 합격케이스랑 match
        sentence5 = ['It matches with your intended focus while it seems coherent with the character elements written in your essay.']
    else: # Intended = 합격케이스랑 안맞음 different. 그리고 detected = 합격케이스랑 안맞음 different.
        sentence5 = ['It does not fully match with your intended focus while it seems incoherent with the character elements written in your essay.']
    
    
    
    # < 문장생성 : Emphasis on You vs. Others >
    emp_sentence1 = ['Compared to the accepted case average for this prompt, you have spent', words_num_few_more,'words to describe the characters in your story.']
    emp_sentence2 = ['In terms of describing yourself, you have utilized', words_num_few_more, 'descriptors compared to the accepted average.']
    emp_sentence3 = ['For this prompt, the accepted students dedicated approximately', self_description_percent, '% of the descriptors for theseves while allotting', other_description_percent, '% for other characters.']
    
    
    # 4th Sentences
    # 내 글에 자기 설명이 accepted avg. 와 비슷한 경우(accepted average와 비슷한 경우 / 오차범위 +-10%)
    # 개인, 그룹의 you관련 사용 값의 분산을 구하고, 평균값으로 나누어서 오차범위를 계산하기
    compare_abs = (abs(your_essay_you_vs_others[0] - admitted_case_avg[0])) # 개인,그룹의 두 값의 차이에 절다값을 적용하여 실질 차이를 계산
    
    #결과확인용 
    print("개인의 에세이 입력값 you:", your_essay_you_vs_others[0])
    print("합격한 학생의 평균값 :", admitted_case_avg[0])
    print("두 값의 절대값 차이 compare_abs", compare_abs)
    print("당신의 에세이에서 절대값을 뺀 수: ", your_essay_you_vs_others[0] - compare_abs)
    print("두 값의 평균값에 +10%적용한 값 : ",  (your_essay_you_vs_others[0] + admitted_case_avg[0])/2 * 0.1)
    print("두 값의 평균값에 -10%적용한 값 : ",  (your_essay_you_vs_others[0] + admitted_case_avg[0])/2 * 0.1)
    
    if (your_essay_you_vs_others[0] - compare_abs) <= your_essay_you_vs_others[0] + (your_essay_you_vs_others[0] + admitted_case_avg[0])/2 * 0.1:  #오차범위 +-10% 이라면, 절대 차이값과 두 값을 더한 평균의 10%를 적용하여 비교계산
        emp_sentence4 = ['It seems that you display an adequate balance in describing yourself and other characters in your essay compared to the accepted average.']
    
    elif (your_essay_you_vs_others[0] - compare_abs) >= your_essay_you_vs_others[0] - (your_essay_you_vs_others[0] + admitted_case_avg[0])/2 * 0.1:
        emp_sentence4 = ['It seems that you display an adequate balance in describing yourself and other characters in your essay compared to the accepted average.']
    
    elif your_essay_you_vs_others[0] > admitted_case_avg[0]: # 내글에 자기 설명이 accepted avg. 보다 더 많은 경우 (accepted average보다 많은 경우/ 오차범위 +10% 이상)
        emp_sentence4 = ['It seems that you may have too much focus on describing other characters (and not enough on you) compared to the essays that worked.']
    
    elif your_essay_you_vs_others[0] < admitted_case_avg[0]: # 내글에 자기 설명이 accepted avg. 보다 더 적은 경우 (accepted average보다 많은 경우/ 오차범위 -10% 이상)
        emp_sentence4 = ['It seems that you may have too much focus on describing yourself (and not enough on other characters) compared to the essays that worked.']
    
    else:
        pass
    
    # Emphasis Character by Section (# of Character Descriptors)
    # Intro / Body 1 / Body 2 / Body 3 / Conclusion (가장 캐릭 설명 많은 Section)
    section = paragraph_divide_ratio(input_text)
    
    emp_char_sec_sentence1 = ['Dividing up the personal statement in 5 equal parts by the word count, the accepted case average indicated that most number of character descriptors are concentrated in the', section[0], 'and', section[1]]
    # Intro / Body 1 / Body 2 / Body 3 / Conclusion (두번째로 캐릭 설명 많은 Section)
    emp_char_sec_sentence2 = section[4]
    
    # admitted_student_for : 문장완성을 위한 값 Focus on Character(s) by Admitted Students for ________
    # result:  #1~3의 결과가 나옴 1: Mostly Me , 2: Me & some others, 3: Other characters
    # sentence1 ~5 : 이것은 문장생성 결과
    
    ## << Chart 표현 부분 >> ##
    # total_character_descriptors_personal:  개인 에세이에서 분석 추출한 총 캐릭터 표현 수
    # descriptors_about_yourself : 개인 에세이 추출 표현 about i
    # total_character_descriptors_group: 1000명의 에세이에서 공통적으로 추출계산한 캐릭터 총 평균값(임의로 정함, 계산후 넣어야 함)
    # descriptors_about_others_group: 1000명의 에세이 추출 others 캐릭터 평균값(임의로 정했음, 계산후 넣어야 함)
    
    ## << Emphasis on You vs. Others >> 그래프 표현 부분 ##
    # admitted_case_avg : ex) [35. 65] 
    # your_essay_you_vs_others : ex) [49, 13] 개인 에세이 계산 결과
    
    # emp_sentence1~4 : Emphasis on You vs. Others의 비교분석값 Sentece 4 커멘트 부분임
    # emp_char_sec_sentence1, emp_char_sec_sentence2 : Emphasis Character by Section (# of Character Descriptors) 문장 생성 부분
    return admitted_student_for, result, sentence1, sentence2, sentence3, sentence4, sentence5, total_character_descriptors_personal,descriptors_about_yourself,total_character_descriptors_group, descriptors_about_others_group, admitted_case_avg, your_essay_you_vs_others, emp_sentence1, emp_sentence2, emp_sentence3, emp_sentence4, emp_char_sec_sentence1, emp_char_sec_sentence2









###### Run ######

# input_text : 입력에세이
# promt_no : 선택질문  >> ex) 'prompt_1'...
# intended_character : mostly me >> 'me' = 1, me & some others : 'meAndOtehrs' = 2, other characters: 'others' = 3
# intended_character의 입력은 'me', 'meAndOtehrs', 'others'


input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""
promt_no = "promt_2"
intended_character = "meAndOtehrs"

result = focusOnCharacters(input_text, promt_no, intended_character)

print(result)

# 결과분석

# admitted_student_for ____ : seletec prompt no. 문장완성을 위한 값 Focus on Character(s) by Admitted Students for ________
# result:  #1~3의 결과가 나옴 1: Mostly Me , 2: Me & some others, 3: Other characters
# sentence1 ~5 : 이것은 문장생성 결과