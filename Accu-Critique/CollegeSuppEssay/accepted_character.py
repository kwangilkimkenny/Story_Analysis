# json file 불러오기

import json
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
from tqdm import tqdm

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

    #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고
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

    #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고
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



def focusOnCharacters(input_text):

    person_num = find_named_persons(input_text)
    charater_num = list(characters(input_text))

    sum_character_num = person_num + charater_num[0] + charater_num[1]
    ratio_i = round((charater_num[0] / sum_character_num),2) * 100

    if ratio_i >= 70: # i 가 70% 이상
        #print("Mostly Me")
        result = 1
    elif 40 <= ratio_i < 70: # i가 40~ 70% 
        #print("Me & some others")
        result = 2
    else:
        #print("Others characters") # i가 40% 이하
        result = 3

    # result : 1~3의 결과가 나옴
    # charater_num[0] : focus on me character nums 
    # charater_num[1] : others character nums
    # peson_num : name이 언급된 캐릭터 수
    return result, charater_num[0], charater_num[1], sum_character_num, person_num


def character_counter_mean():
    character_cnt = [] # nums of character 'focus on you'
    character_all_cnt = [] # total nums of character
    character_others = []
    character_name = []
    path = "./data/accepted_data/ps_essay_evaluated.csv"
    data = pd.read_csv(path)
    #Score를 인덱스로 변환하여 데이터 찾아보기
    data.set_index('Score', inplace=True)
    for i in tqdm(data.index):
        if i is not None and i >= 4:
            get_essay = data.loc[i, 'Essay']

            input_ps_essay = get_essay
            re = focusOnCharacters(str(input_ps_essay))
            result = re[1]
            character_cnt.append(result)
            character_all_cnt.append(re[2])

    #평균값 구하기
    # focus on me
    accepted_character_focus_on_me_mean = round(sum(character_cnt) / len(character_cnt), 1)
    accepted_character_all_mean = round(sum(character_all_cnt) / len(character_all_cnt), 1)
    #accepted_character_others_mean = round(sum(character_others) / len(character_others), 1)
    #accepted_character_name_all_mean = round(sum(character_name) / len(character_name), 1)

    result_data = {
        "합격한 학생들의 편균값(에세이에서 '나'에 대한 표현 총 수)" :accepted_character_focus_on_me_mean,
        "합격한 학생들의 전체 문장에 대한 캐릭터 총 표현 비율" : accepted_character_all_mean
    }
    return result_data


# def character_counter_mean():
#     character_cnt = []
#     for i in range(100, 300): #5 -> 978
#         path = "./data/accepted_data/" + "each_personal_essay_" + str(i) + ".json"
#         # ./data/accepted_data/each_personal_essay_0.json
#         with open(path) as f:
#             data = json.load(f)

#             input_ps_essay = data["data"]
#             re = focusOnCharacters(str(input_ps_essay))
#             result = re[1]
#             character_cnt.append(result)

#     #평균값 구하기
#     accepted_character_mean = round(sum(character_cnt) / len(character_cnt), 1)
    
#     return accepted_character_mean


### run ###
print('result :', character_counter_mean())

