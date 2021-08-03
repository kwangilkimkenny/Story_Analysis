# updating 20210802


# Focus on Character 분석 : PPT 10~12 pages 내용을 코드로 구현한 거임
# 문장을 입력하면 Character part의 모든 분석이 가능


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

from collections import Counter


import itertools
    
nlp = spacy.load('en_core_web_lg')


##########  key_value_print
##########  key: value 형식으로 나뉨 
def key_value_print (dictonrytemp) : 
    print("#"*100)
    for key in dictonrytemp.keys() : 
        print(key,": ",dictonrytemp[key])
        print()
    print("#"*100)
    
def character_to_convert (number) : 
    
    character_str = "me"
    
    if number == 1 : 
        
        character_str = "me"
        
    elif number == 2 : 
        
        character_str = "meAndOtehrs"
        
    elif number == 3 : 
        
         character_str = "others"
         
    return character_str 
    
    

# 이름으로 인물 수 계산
def find_named_persons(text):
    # Create Doc object
    doc2 = nlp(text)
    # Identify the persons
    persons = [ent.text for ent in doc2.ents if ent.label_ == 'PERSON']
    print("############## persons:", persons)
    
    #총 인물 수
    get_person = len(persons)
    # Return persons
    return get_person,persons


def characters(input_text):   #### Character 찾기 
    #소문자로 변환
    input_lower_text = input_text.lower()
    about_doc = nlp(input_text)
    print("about_doc:",about_doc)
    token_list = {}
    for token in about_doc:
        #print (token, token.idx)
        token_list.setdefault(token, token.idx)
    
    li_doc = list(token_list.keys())

    #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고
    i_character_list = ['I', 'my', 'me', 'mine','myself']
    
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

    #### 결과값 해석 ####
    # get_i : i에 관련한 캐릭터 표현 단어의 총 수
    # get_others : others에 관련한 캐릭터 표현 단어의 총 수(except for 'i')
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
    i_character_list = ['i', 'my', 'me', 'mine','myself']
    
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
                    'yours', 'he','him','his' 'she','her','it','someone','their', 'aunt',
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
    
    print("#"*200)
    print("get_others_words:",get_others_words)
    print("get_i_words:",get_i_words)
    print("#"*200)
    
    ########  i ---> I 로 치환 ############# 
    get_I_words = []
    for i in get_i_words : 
        temp = i.replace('i', "I")
        get_I_words.append(temp)
        
    
    # print("get_i_words:",get_I_words)

    return get_I_words, get_others_words


# Number of Characters
def NumberofCharacters(text):
    ################################################################################################################
    # # printX("n_of_ch step 1");
    essay_input_corpus = str(text) #문장입력
    essay_input_corpus = essay_input_corpus.lower()#소문자 변환

    sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화
    total_sentences = len(sentences)#토큰으로 처리된 총 문장 수
    total_words = len(word_tokenize(essay_input_corpus))# 총 단어수
    split_sentences = []
    ################################################################################################################
    # # printX("n_of_ch step 2");
    for sentence in sentences:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences.append(words)

    ################################################################################################################
    # # printX("n_of_ch step 3");
    skip_gram = 1
    workers = multiprocessing.cpu_count()
    ################################################################################################################
    # # printX2("n_of_ch step 3.1",workers);
    bigram_transformer = Phrases(split_sentences)
    ################################################################################################################
    # # printX("n_of_ch step 4");

    model = gensim.models.word2vec.Word2Vec(bigram_transformer[split_sentences], workers=workers, sg=skip_gram, min_count=1)
    ################################################################################################################
    # # printX("n_of_ch step 5");

    model.train(split_sentences, total_examples=sum([len(sentence) for sentence in sentences]), epochs=100)
    ################################################################################################################
    # # printX("n_of_ch step 6");
    
    #모델 설계 완료

    #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고
    character_list = ['i', 'my', 'me', 'mine', 'you', 'your', 'they','them',
                    'yours', 'he','him','his' 'she','her','it','someone','their', 'myself', 'aunt',
                    'brother','cousin','daughter','father','grandchild','granddaughter','granddson','grandfather',
                    'grandmother','great-grandchild','husband','ex-husband','son-in-law', 'daughter-in-law','mother',
                    'niece','nephew','parents','sister','son','stepfather','stepmother','stepdaughter', 'stepson',
                    'twin','uncle','widow','widower','wife','ex-wife','aunt',
                    'baby', 'beget', 'brother', 'buddy', 'conserve', 'counterpart', 'cousin', 'daughter', 'duplicate', 'ex',
                    'father', 'forefather', 'founder', 'gemini', 'grandchild', 'granddaughter', 'grandfather', 'grandma', 'he', 'helium',
                    'husband', 'i', 'in', 'iodine', 'law', 'maine', 'match', 'mine', 'mother', 'nephew', 'niece', 'one', 'parent', 'person',
                    'rear', 'sister', 'son', 'stepdaughter', 'stepfather', 'stepmother', 'stepson', 'twin', 'uncle', 'widow', 'widower', 'wife']
    
    ####문장에 char_list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
    #우선 토큰화한다.
    retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
    token_input_text = retokenize.tokenize(essay_input_corpus)
    ### printX(token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
    #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
    ################################################################################################################
    # # printX("n_of_ch step 7");
    filtered_chr_text = []
    for k in token_input_text:
        for j in character_list:
            if k == j:
                filtered_chr_text.append(j)
    
    ################################################################################################################
    # # printX("n_of_ch step 8");
    ### printX2 (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
    
    # filtered_chr_text_ = set(filtered_chr_text) #중복제거
    filtered_chr_text_ = filtered_chr_text #중복제거하지 않음(모든 캐릭터 수를 카운드)
    filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
    ## printX(filtered_chr_text__) # 중복값 제거 확인
    
    # for i in filtered_chr_text__:
    #     ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
    
    char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 캐릭터 표현 수
    char_count_ = len(filtered_chr_text__) #캐릭터 표현 총 수
    ################################################################################################################
    # # printX("n_of_ch step 9");
    
    ################################################################################################################
    # # printX2("중복제거된 캐릭터 표현 수:", char_count_)
    
    #전체 단어중에서 캐릭터 관련 단어가 얼마나 포함되었는지 비율 계산, 그 비율에 맞게 결과값 조정
    result_char_ratio = round(char_total_count/total_words * 100, 2)
    ################################################################################################################
    # # printX2("전체 단어중에서 캐릭터 관련 단어가 얼마나 포함되었는지 비율:", result_char_ratio)
    # 전체 단어중에서 캐릭터 관련 단어가 얼마나 포함되었는지 비율: 12.36
    
    ###################################
    ########### 추가한 코드 ##############
    ###################################
    #이름을 인식하여 캐릭터 수 카운트하기
    doc = nlp(text)
    ext_label_char =[(X.text, X.label_) for X in doc.ents] #레이블 추출
    # ext_label_char = set(ext_label_char) #같은 이름 중복 제거
    ext_label_char = ext_label_char #같은 이름 중복 제거하지 않음(모든 캐릭터 수를 카운트 할 경우)
    # print("get lavel of entity:", ext_label_char)
    ################################################################################################################
    # # printX2("레이블 추출:", ext_label_char)
    ext_char_name = []
    cont = 0 #캐릭터 수 초기화
    for i in ext_label_char:
        if 'PERSON' in i: # PERSON이 튜플 안에 있다면 캐릭터니까 카운트를 한다. 
            ext_char_name.append(i[0])
            cont += 1
        
    ################################################################################################################
    # # printX2("에세이에 포함된 이름 수:", cont)
    #총 캐릭터수 계산(character_list + cont) 합치면
    char_count_ = char_total_count + cont
    char_count_ratio = round(char_count_ / total_words * 100, 2) #전체문장길이 캐릭터 비율 적용
    all_char_name_list = filtered_chr_text + ext_char_name
    ################################################################################################################
    # # printX2("1명 에세이 총 캐릭터수:", char_count_)

    esy_sent_all = []
    for w in sentences:
        esy_sent = []
        w_t = word_tokenize(w)
        esy_sent.append(w)
        esy_sent.append(w_t)
        esy_sent_all.append(esy_sent)


    emp_on_me_list = []
    for cmp in esy_sent_all:
        # print('cmp:', cmp[1])
        for cmp_ in cmp:
            #print('cmp_:', cmp_)
            for chk in cmp_:
                if chk in filtered_chr_text:
                    cmp_capitalize = cmp[0].capitalize()
                    emp_on_me_list.append(cmp_capitalize)

    character_on_sentences_all = list(set(emp_on_me_list))
    # character 포함된 문장 추출결과
    print("----------------------------------------------------")
    # print('character_on_sentences_all:', character_on_sentences_all)


    # character_on_sentences_all : 캐릭터 관련 단어가 포함된 모든 문장 추출
    # filtered_chr_text : character 직점 관련 단어 'my', 'i', 'i', 'my', 'my', 'i', '...
    # ext_char_name : 이름 추출
    # all_char_name_list : 이름 + 캐릭터 표현 대명사 모두 추출 ===========> 웹에 표시할 것
    # char_count_ratio : 전체 단어중에서 이름 관련 단어가 얼마나 포함되었는지 비율

    return character_on_sentences_all, filtered_chr_text, ext_char_name, all_char_name_list, char_count_ratio


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
    
    charater_words=  character_words(text)    
    indicator_describing_yourself = charater_words[0]
    indicator_describing_others = charater_words[1]
    


    charater_words = list(charater_words)
    charater_words = sum(charater_words, [])
    charater_words = set(charater_words) #중복제거
    charater_words = list(charater_words)
    
    print('charater_words: ', charater_words)
    print('len:', len(charater_words))


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
    
    return f_result, f_result_2n, result_, group_parts_mean, sentence_2nd_emp_sec ,indicator_describing_yourself, indicator_describing_others



from gensim import corpora, models, similarities

#질문 7개
def get_appropriate_pmt_by_pmt_nomber_seven(input_text):
    
    documents = ["Some students have a background, identity, interest, or talent that is so meaningful they believe their application would be incomplete without it. If this sounds like you, then please share your story.",
                "The lessons we take from obstacles we encounter can be fundamental to later success. Recount a time when you faced a challenge, setback, or failure. How did it affect you, and what did you learn from the experience?",
                "Reflect on a time when you questioned or challenged a belief or idea. What prompted your thinking? What was the outcome?",
                "Describe a problem you've solved or a problem you'd like to solve. It can be an intellectual challenge, a research query, an ethical dilemma - anything that is of personal importance, no matter the scale. Explain its significance to you and what steps you took or could be taken to identify a solution.",
                "Discuss an accomplishment, event, or realization that sparked a period of personal growth and a new understanding of yourself or others.",
                "Describe a topic, idea, or concept you find so engaging that it makes you lose all track of time. Why does it captivate you? What or who do you turn to when you want to learn more? ",
                "Share an essay on any topic of your choice. It can be one you've already written, one that responds to a different prompt, or one of your own design."]


  
    # remove common words and tokenize them
    stoplist = set('for a of the and to in'.split())

    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

    # remove words those appear only once
    all_tokens = sum(texts, [])

    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) ==1)
    texts = [[word for word in text if word not in tokens_once]
            for text in texts]
    dictionary = corpora.Dictionary(texts)

    dictionary.save('deerwester.dict')  # save as binary file at the dictionary at local directory
    dictionary.save_as_text('deerwester_text.dict')  # save as text file at the local directory



    #input answer
    text_input = input_text #문장입력....
    #text_input = """I inhale deeply and blow harder than I thought possible, pushing the tiny ember from its resting place on the candle out into the air. The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

    new_vec = dictionary.doc2bow(text_input.lower().split()) # return "word-ID : Frequency of appearance""
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('deerwester.mm', corpus) # save corpus at local directory
    corpus = corpora.MmCorpus('deerwester.mm') # try to load the saved corpus from local
    dictionary = corpora.Dictionary.load('deerwester.dict') # try to load saved dic.from local
    tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
    corpus_tfidf = tfidf[corpus]  # map corpus object into tfidf space
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize LSI
    corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus
    topic = lsi.print_topics(2)
    lsi.save('model.lsi')  # save output model at local directory
    lsi = models.LsiModel.load('model.lsi') # try to load above saved model

    doc = text_input

    vec_bow = dictionary.doc2bow(doc.lower().split())  # put newly obtained document to existing dictionary object
    vec_lsi = lsi[vec_bow] # convert new document (henceforth, call it "query") to LSI space
    index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and indexize it
    index.save('deerwester.index') # save index object at local directory
    index = similarities.MatrixSimilarity.load('deerwester.index')
    sims = index[vec_lsi] # calculate degree of similarity of the query to existing corpus

    print(list(enumerate(sims))) # output (document_number , document similarity)

    sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sort output object as per similarity ( largest similarity document comes first )
    print(sims) # 가장 질문에 대한 답변이 적합한 순서대로 출력
    
    # result_sims = []
    
    quada_list = []
    
    for temp in sims : 
        
        #quada_list.append(round(float(temp[1]),3))
        
        quada_list.append([temp[0],round(float(temp[1]),3)])

    return quada_list

# input_text : 입력에세이
# promt_no : 선택질문  >> ex) 'prompt_1'...
# intended_character : mostly me >> 'me' = 1, me & some others : 'meAndOtehrs' = 2, other characters: 'others' = 3
# intended_character의 입력은 'me', 'meAndOtehrs', 'others'

def focusOnCharacters(input_text, promt_no, intended_character):

    person_num = find_named_persons(input_text)[0]
    indicator_all_character_descriptors = find_named_persons(input_text)[1]
    
    charater_num = list(characters(input_text))
    print('character_num:', charater_num)
    
    # ppt >> Number of Charactr Descriptors(graph로 표현해야 하는 값)
    total_character_descriptors_personal = sum(charater_num) # 개인 에세이에서 분석 추출한 총 캐릭터 표현 수
    descriptors_about_yourself = charater_num[0] #개인 에세이 추출 표현 about i
    
    print("#"*200)
    print("total_character_descriptors_personal:",total_character_descriptors_personal)
    print("descriptors_about_yourself:",descriptors_about_yourself)
    print("#"*200)
    
    total_character_descriptors_group = 40 ####### 1000명의 에세이에서 공통적으로 추출계산한 캐릭터 총 평균값(임의로 정함, 계산후 넣어야 함)
    descriptors_about_others_group = 10 ###### 1000명의 에세이 추출 others 캐릭터 평균값(임의로 정했음, 계산후 넣어야 함)
    
    ##################################################################################################################
    # ppt >> Emphasis on You vs. Others
    admitted_case_avg = [35, 65] # you(I, me) : 35%, others: 65% >>>>>>>> 합격한 학생들의 평균갑으로 임의로 넣음(나중에 계산해서 넣어야 함)
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

    self_description_percent = 0
    other_description_percent = 0 
    
    try : 
        # i, others 관련 단어 사용 비율 계산하여 3nd  >> emp_sentence3
        self_description_percent = [round((your_essay_you_vs_others[0] / your_essay_you_vs_others[1]) * 100)]

        other_description_percent = [round((your_essay_you_vs_others[1] / your_essay_you_vs_others[0]) * 100)]
    
    except : 
        pass 
    
    # Number of Character Descriptors
    numb_of_all_chars = NumberofCharacters(input_text)
    print("number of all characters:", numb_of_all_chars)
    # result_char_ratio : 전체 단어중에서 캐릭터 관련 단어가 얼마나 포함되었는지 비율
    print("전체 단어중에서 캐릭터 관련 단어가 얼마나 포함되었는지 비율", numb_of_all_chars[1])


    # 새로 추가된 부분 - Prompt 별로 캐릭터 표현 구분 출력 (multiple characters - i포함해서 4명이 한 문장에 등장하는 것)
    
    pmpt_1 = {'Mostly me': 29, 'Me and a few others':59, 'Multiple characters':23, 'Emphasis on me':72, 'Emphasis on others': 28}
    pmpt_2 = {'Mostly me': 28, 'Me and a few others':61, 'Multiple characters':11, 'Emphasis on me':64, 'Emphasis on others': 36}
    pmpt_3 = {'Mostly me': 39, 'Me and a few others':53, 'Multiple characters':8, 'Emphasis on me':69, 'Emphasis on others': 31}
    pmpt_4 = {'Mostly me': 19, 'Me and a few others':52, 'Multiple characters':29, 'Emphasis on me':61, 'Emphasis on others': 39}
    pmpt_5 = {'Mostly me': 30, 'Me and a few others':58, 'Multiple characters':12, 'Emphasis on me':66, 'Emphasis on others': 34}
    pmpt_6 = {'Mostly me': 41, 'Me and a few others':45, 'Multiple characters':14, 'Emphasis on me':70, 'Emphasis on others': 30}

############################################################################################### 

    # *Prompt #7은 자기 에세이와 가장 가까운 prompt를 1-6번 안에서 고르라고 한걸로 기억해요 - 이것을 자동으로 계산해주기
    # 이 부분을 계산하기 위해서는, 에세이와 가장 가까운 prompt의 속성을 자동으로 1~6번에서 골라줘야함
    # 유사문장비교 기술을 이용하여 가장 관련성이 높은 prompt를 선택하게 해야함
    def selected_seven(prompt_no):
        if(prompt_no == "ques_seven") : # (수정 후) 7번의 Prompt를 선택했고, 에세이를 입력했다면,
            re_prompt_no_analysis = get_appropriate_pmt_by_pmt_nomber_seven(input_text)
            print('re_prompt_no_analysis:', re_prompt_no_analysis[0][0]) # 가장 일치율이 높은 prompt를 자동으로 찾아줌
            most_sim_prompt_no_by_selected_seven = re_prompt_no_analysis[0][0]
            if most_sim_prompt_no_by_selected_seven == 0:
                prompt_no == "ques_one"
            elif most_sim_prompt_no_by_selected_seven == 1:
                prompt_no == "ques_two"
            elif most_sim_prompt_no_by_selected_seven == 3:
                prompt_no == "ques_three"
            elif most_sim_prompt_no_by_selected_seven == 4:
                prompt_no == "ques_four"
            elif most_sim_prompt_no_by_selected_seven == 5:
                prompt_no == "ques_five"
            else: # most_sim_prompt_no_by_selected_seven == 6:
                prompt_no == "ques_six"

        print('selected_prompt no:', prompt_no)
        return prompt_no

############################################################################################### 

    
    # input prompt number. 결과물 seleected_prompt_number을 sentece4에 적용
    selected_prompt_number = []
    if promt_no == "ques_one":
        selected_prompt_number.append("prompt #.1")
        web_result_prompt_by_selected = pmpt_1
    elif promt_no == "ques_two":
        selected_prompt_number.append("prompt #.2")
        web_result_prompt_by_selected = pmpt_2
    elif promt_no == "ques_three":
        selected_prompt_number.append("prompt #.3")
        web_result_prompt_by_selected = pmpt_3
    elif promt_no == "ques_four":
        selected_prompt_number.append("prompt #.4")
        web_result_prompt_by_selected = pmpt_4
    elif promt_no == "ques_five":
        selected_prompt_number.append("prompt #.5")
        web_result_prompt_by_selected = pmpt_5
    elif promt_no == "ques_six":
        selected_prompt_number.append("prompt #.6")
        web_result_prompt_by_selected = pmpt_6
    elif promt_no == "ques_seven":
        selected_prompt_number.append("prompt #.7")
        # prompt 1~6 중에서 선택한 에세이와 가장 가까운 prompt에 해당하는 질문을 자동 선택하는 코드
        web_result_prompt_by_selected = selected_seven("ques_seven")
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
        sentence3 = ['If you wish to shift the essay’s direction towards your original intention, you may consider reducing the number of characters. You may focus more on your own thoughts and interaction between a few core characters in the story.']
    elif intended_character_ == 3: #Other characters
        sentence3 = ['If you wish to shift the essay’s direction towards your original intention, you may consider increasing the number of characters and including more interactions between members.']
    else:
        pass

    ################################
    # ---- 합격케이스의 평균값 입력 ---- #
    adm_mean_result = 2 # 합격케이스 평균값 설정(이것은 1000명의 통계를 돌려서 결과를 반영해야함. 지금은 임의값 적용하였음)
    
    #비교항목
    admit_mean_value_mostly_me = 1 # 합격케이스 평균값 설정(이것은 1000명의 통계를 돌려서 결과를 반영해야함. 지금은 임의값 적용하였음)
    admit_mean_value_me_and_a_few_others = 2 # 합격케이스 평균값 설정(이것은 1000명의 통계를 돌려서 결과를 반영해야함. 지금은 임의값 적용하였음)
    admit_mean_value_multiple_characters = 3 # 합격케이스 평균값 설정(이것은 1000명의 통계를 돌려서 결과를 반영해야함. 지금은 임의값 적용하였음)
    


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
    
    indicator_describing_yourself  = section[5]
    indicator_describing_others  = section[6]
    
    
    emp_char_sec_sentence1 = ['Dividing up the personal statement in 5 equal parts by the word count, the accepted case average indicated that most number of character descriptors are concentrated in the', section[0], 'and', section[1]]
    # Intro / Body 1 / Body 2 / Body 3 / Conclusion (두번째로 캐릭 설명 많은 Section)
    emp_char_sec_sentence2 = section[4]
    

    emphasis_character_all = section[2]
    emphasis_character_your_essay = section[3]

    # print("#"*200)
    # print("emphasis_character_all:", emphasis_character_all)
    # print("emphasis_character_your_essay:",emphasis_character_your_essay)
    # print("#"*200)
    # <<<<<< 결과 해석 >>>>>>> #

    # admitted_student_for : 문장완성을 위한 값 'Focus on Character(s) by Admitted Students for _선택한 Prompt # 문항_'
    # intended_character :  1~3의 결과가 나옴 1: Mostly Me , 2: Me & some others, 3: Other characters
    # result:  'Detected characters from essay' 1~3의 결과가 나옴 1: Mostly Me , 2: Me & some others, 3: Other characters
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
    
    
    
    
    data = {
        
        "admitted_student_for": admitted_student_for,
        
        "intended_character": character_to_convert(intended_character_),
        "detected_character": character_to_convert(result),
        
        "focus_all_comment_1": sentence1,
        "focus_all_comment_2": sentence2,
        "focus_all_comment_3": sentence3,
        "focus_all_comment_4": sentence4,
        "focus_all_comment_5": sentence5,
        
        "total_character_descriptors_personal": total_character_descriptors_personal,
        "descriptors_about_yourself": descriptors_about_yourself,
        "total_character_descriptors_group": total_character_descriptors_group,
        "descriptors_about_others_group": descriptors_about_others_group,
        "admitted_case_avg": admitted_case_avg,
        "your_essay_you_vs_others": your_essay_you_vs_others,
        
        "emphasis_all_comment_1": emp_sentence1,
        "emphasis_all_comment_2": emp_sentence2,
        "emphasis_all_comment_3": emp_sentence3,
        "emphasis_all_comment_4": emp_sentence4,
        
        "emphasis_character_by_section_comment_1": emp_char_sec_sentence1,
        "emphasis_character_by_section_comment_2": emp_char_sec_sentence2,
        "emphasis_character_all": emphasis_character_all,
        "emphasis_character_your_essay": emphasis_character_your_essay,
        
        "indicator_all_character_descriptors" : dict(Counter(indicator_all_character_descriptors)),
        "indicator_describing_yourself" : dict(Counter(indicator_describing_yourself)),
        "indicator_describing_others" : dict(Counter(indicator_describing_others)),

        
        # web_result_prompt_by_selected 결과로 웹사이트에 표시하는 부분,선택한 Prompt에 해당하는 합격생들의 이상적인 값 표시로 고정값
        # 예) 이하 5가지로 문항별 고정값을 웹페이지에 표시한다. 개인 에세이 분석 결과값은 위에서 계산한 값을 표시해야 함
        # Mostly me 29%
        # Me and a few others 59%
        # Multiple characters 23%
        # Emphasis on me: 72%
        # Emphasis on others: 28%

        "web_result_prompt_by_selected" : web_result_prompt_by_selected 
        
        
    }
    
    return data 




#######################################################
# input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""
# input_text = "Reflect on a time when you challenged a belief or idea. What prompted you to act? Would you make the same decision again? My shaking fingers closed around the shiny gold pieces of the saxophone in its case, leaving a streak of fingerprints down the newly cleaned exterior. The soft and melodic sound of a classic jazz ballad floated out of a set of speakers to my right, mixing with the confident chatter of students in the back row. As usual, I shuffled around in the back of the classroom, attempting to blend in with the sets of cubbies. With my confidence already fading, I sat and thought to myself, I wonder if it’s too late to drop this course? I heard Art Exploration still has plenty of openings! The director stepped to the front of the room, and snapped his fingers in a slow rhythm that the drummer tapped out with his worn wooden sticks. The band congregated in the middle of the room, each person tapping his or her feet along with the drummer. Just relax! I scolded myself, just be jazzy and no one will notice you look out of place. I attempted to mimic the laid-back, grooving movements of my peers, but to no avail. My movements were about as far away from ‘jazzy’ as one could possibly get. I was used to the rigid accents and staccatos of the concert band world. As the solo section began to move around the room most students seemed relaxed and loose, acting as if they were easygoing musicians on a street corner in New Orleans. I felt my body tense and my eyes dart nervously around the room for a quick escape route. My confidence plummeted as the person in front of me swung a closing riff and looked expectantly in my direction. I quickly pressed the mouthpiece to my tongue and played a few stiff sounding riffs, but the notes seemed to fall flat. I felt my face flush a nice shade of deep red as I stood in the middle of the room with the drummer tapping away, waiting for me to step in and finish. All of a sudden, the director shouted, ‘Play a riff! Dance! Come on, Phoebe, at least do something!’ His exasperated tone prompted me to attempt to salvage the bleak looking situation I was currently facing."
# input_text = "When it comes to applying to university in the United States, it’s important to note that the Common App isn’t the only game in town.\
# The Coalition Application is a newer (and smaller) application platform, through which students can apply to over 100 US colleges, both state and private. Like the Common App, students can submit their applications to multiple colleges at once. \
# Specifically, students can use the Coalition Application to:\
# Submit their basic information, like name and address, school grades and GPA and relevant test scores. \
# Store relevant information in the Coalition Application ‘Locker’ that allows students to keep all their application materials in one place.\
# Keep supporting audio and visual files that might help their application to stand out from the crowd.\
# Note: The Coalition Application might be a more suitable application platform for students interested in arts and creative subjects. The ability to submit audio/visual material means that a US college applicant can really showcase their creative portfolio. \
# Like the Common App, the Coalition Application asks students to submit an application essay, that is then sent to all of their prospective colleges. And, also like the Common App, students must respond to one of several essay prompts that are designed to give universities a comprehensive picture of who they are, what drives them, and their ambitions for university and beyond.\
# Recently, Coalition for College released the essay prompts for this year’s Coalition Application Essay.\
# What are the Coalition Application Essay Prompts? \
# For the 2019-20 application cycle, Coalition for College are asking applicants to answer one of the following five questions. \
# Tell a story from your life, describing an experience that either demonstrates your character or helped to shape it. \
# Describe a time when you made a meaningful contribution to others in which the greater good was your focus. Discuss the challenges and rewards of making your contribution\
# Has there been a time when you’ve had a long-cherished or accepted belief challenged? How did you respond? How did the challenge affect your beliefs?\
# What is the hardest part of being a teenager now? What’s the best part? What advice would you give a younger sibling or friend (assuming they would listen to you)?\
# Submit an essay on a topic of your choice.\
# Each of the prompts above are suitably open ended, and designed to give students the space and the flexibility to fully showcase their positive personal and academic qualities. \
# But the freedom and flexibility of five such open prompts (particularly no.5!) also bring with them certain drawbacks. Students might come to their guidance sessions with you and ask how they’re supposed to brainstorm and plan a response to such open-ended questions. \
# Some of the prompts listed above will make students think (wrongly) that they need to have lived the life of an action hero, or be a tech entrepreneur before the age of 15. Nothing could be further from the truth! \
# In the next section, we’ll look at some of the techniques for planning the Coalition Application essay before writing it. "


# Personal Essay Sample #1 (Prompt #1이나 Prompt #4에 맞는 아름다운 에세이)

input_text = """My hand lingered on the cold metal doorknob. I closed my eyes as the Vancouver breeze ran its chilling fingers through my hair. The man I was about to meet was infamous for demanding perfection. But the beguiling music that faintly fluttered past the unlatched window’s curtain drew me forward, inviting me to cross the threshold. Stepping into the apartment, under the watchful gaze of an emerald-eyed cat portrait, I entered the sweeping B Major scale.

Led by my intrinsic attraction towards music, coupled with the textured layers erupting the instant my fingers grazed the ivory keys, driving the hammers to shoot vibrations up in the air all around me, I soon fell in love with this new extension of my body and mind. My mom began to notice my aptitude for piano when I began returning home with trophies in my arms. These precious experiences fueled my conviction as a rising musician, but despite my confidence, I felt like something was missing.

Back in the drafty apartment, I smiled nervously and walked towards the piano from which the music emanated. Ian Parker, my new piano teacher, eyes-closed and dressed in black glided his hands effortlessly across the keys. I stood beside a leather chair, waiting as he finished the phrase. He stood up. I sat down.

Chopin Black Key Etude — a piece I knew so well I could play it eyes-closed. I took a breath and positioned my right hand in a G-flat 2nd inversion. 
Just one measure in, I was stopped. 
	“Start again.”
	Taken by surprise, I spun left. His eyes were on the score, not me. 
	I started again. Past the first measure, first phrase, then stopped again. What is going on? 
	
	“Are you listening?”
I nodded. Of course I am. 
“But are you really listening?”

As we slowly dissected each measure, I felt my confidence slip away. The piece was being chipped into fragments. Unlike my previous teachers, who listened to a full performance before giving critical feedback, Ian stopped me every five seconds. One hour later, we only got through half a page. 

Each consecutive week, the same thing happened. I struggled to meet his expectations. 
“I’m not here to teach you just how to play. I’m here to teach you how to listen.” 
I realized what Ian meant — listening involves taking what we hear and asking: is this the sound I want? What story am I telling through my interpretation? 

Absorbed in the music, I allowed my instincts and muscle memory to take over, flying past the broken tritones or neapolitan chords. But even if I was playing the right notes, it didn’t matter. Becoming immersed in the cascading arpeggio waterfalls, thundering basses, and fairydust trills was actually the easy part, which brought me joy and fueled my love for music in the first place. However, music is not just about me. True artists perform for their audience, and to bring them the same joy, to turn playing into magic-making, they must listen as the audience. 

The lesson Ian taught me echoes beyond practice rooms and concert halls. I’ve learned to listen as I explore the hidden dialogue between voices, to pauses and silence, equally as powerful as words. Listening is performing as a soloist backed up by an orchestra. Listening is calmly responding during heated debates and being the last to speak in a SPS Harkness discussion. It’s even bouncing jokes around the dining table with family. I’ve grown to envision how my voice will impact the stories of those listening to me.

To this day, my lessons with Ian continue to be tough, consisting of 80% discussion and 20% playing. When we were both so immersed in the music that I managed to get to the end of the piece before he looked up to say, “Bravo.” Now, even when I practice piano alone, I repeat my refrain: Are you listening?  """



promt_no = "ques_one"
intended_character = "meAndOtehrs"

data = focusOnCharacters(input_text, promt_no, intended_character)

key_value_print(data)

####################################################################################################
# admitted_student_for :  ['prompt #.1']

# intended_character :  others
# detected_character :  me

# focus_all_comment_1 :  ['Regarding the number of characters, you’ve intended to focus on yourself mainly.']
# focus_all_comment_2 :  ['The AI analysis indicates that your personal statement seems to be focusing mostly on you.']
# focus_all_comment_3 :  ['If you wish to shift the essay’s direction towards your original intention, you may consider increasing the number of characters and including more interactions between members.']
# focus_all_comment_4 :  ['The admitted case trend indicates many applicants tend to focus on themselves and a few other core characters for', ['prompt #.1']]
# focus_all_comment_5 :  ['It does not fully match with your intended focus while it seems incoherent with the character elements written in your essay.']

# total_character_descriptors_personal :  62
# descriptors_about_yourself :  49

# total_character_descriptors_group :  40
# descriptors_about_others_group :  10

# admitted_case_avg :  [35, 65]
# your_essay_you_vs_others :  [49, 13]

# emphasis_all_comment_1 :  ['Compared to the accepted case average for this prompt, you have spent', [14, 'more'], 'words to describe the characters in your story.']
# emphasis_all_comment_2 :  ['In terms of describing yourself, you have utilized', [14, 'more'], 'descriptors compared to the accepted average.']
# emphasis_all_comment_3 :  ['For this prompt, the accepted students dedicated approximately', [377], '% of the descriptors for theseves while allotting', [27], '% for other characters.']
# emphasis_all_comment_4 :  ['It seems that you display an adequate balance in describing yourself and other characters in your essay compared to the accepted average.']

# emphasis_character_by_section_comment_1 :  ['Dividing up the personal statement in 5 equal parts by the word count, the accepted case average indicated that most number of character descriptors are concentrated in the', 'conclusion', 'and', 'body #3']
# emphasis_character_by_section_comment_2 :  ['Comparing this with your essay, we see a different pattern.']

# emphasis_character_all :  [9, 7, 6, 4, 3]
# emphasis_character_your_essay :  [3, 4, 7, 8, 8]

# indicator_all_character_descriptors :  ['Debussy', 'Francesca', 'Francesca', 'Francesca']
# indicator_describing_yourself :  {'me', 'i', 'myself', 'my'}
# indicator_describing_others :  {'her', 'their', 'it', 'someone'}

####################################################################################################





# 결과해석

# admitted_student_for : 문장완성을 위한 값 'Focus on Character(s) by Admitted Students for _선택한 Prompt # 문항_'
# intended_character :  1~3의 결과가 나옴 1: Mostly Me , 2: Me & some others, 3: Other characters
# result:  'Detected characters from essay' 1~3의 결과가 나옴 1: Mostly Me , 2: Me & some others, 3: Other characters
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
    