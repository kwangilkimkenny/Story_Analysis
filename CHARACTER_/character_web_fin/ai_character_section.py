## 캐릭터 분석 단독 페이지를 구현한 코드 ##

# MBTI를 제외한 4개의 분석이 가능함
# 최종점수 계산이 가능함
# 1000명의 데이터를 적용해서 평균값을 비교완료 - 계산결과는 이미 결과 도출하여 고정값으로 계산결과를 적용하였음.(나중에 자동 업데이트 필요함, 현재는 전체평균값 수동입력)


## 결과임  ## lacking : 0 , ideal : 1, overboard : 2
# 최종결과 ex)  [1, 0, 2, 0, 2.75]
# Character Descriptiveness: 1  ideal 
# Number of Characters : 0 lacking
# Emphasis on You : 2 overboard
# Emphasis on Others : 0 lacking
# Overall Character Rating : 2.75

# ===============================================================
# Character Descriptiveness :  67.0
# ===============================================================
# ['person', 'i', 'them', 'her', 'you', 'they', 'me', 'one', 'myself', 'their', 'in', 'it', 'my']
# ai_character_section.py:128: DeprecationWarning: Call to deprecated `most_similar_cosmul` (Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead).
#   ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
# =============================================
# Number of Characters : 145
# =============================================
# ['i', 'me', 'one', 'my']
# ai_character_section.py:379: DeprecationWarning: Call to deprecated `most_similar_cosmul` (Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead).
#   ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
# =============================================
# Emphasis on You : 66
# =============================================
# ['person', 'them', 'her', 'they', 'myself', 'their', 'in', 'it']
# ai_character_section.py:455: DeprecationWarning: Call to deprecated `most_similar_cosmul` (Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead).
#   ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
# =============================================
# Emphasis on Others : 33
# =============================================
# 1명의 에세이 결과 계산점수 : (67.0, 145, 66, 33)
# min_ 30
# max_:  123
# div_: 30
# cal_abs 절대값 : 10.0
# compare7 : 24.0
# compare6 : 28.8
# compare5 : 36.0
# compare4 : 48.0
# compare3 : 72.0
# Ideal: 1
# min_ 191
# max_:  764
# div_: 191
# cal_abs 절대값 : 333
# compare7 : 103.83333333333333
# compare6 : 124.6
# compare5 : 155.75
# compare4 : 207.66666666666666
# compare3 : 311.5
# Lacking: 2
# min_ 5
# max_:  22
# div_: 5
# cal_abs 절대값 : 52
# compare7 : 13.333333333333334
# compare6 : 16.0
# compare5 : 20.0
# compare4 : 26.666666666666668
# compare3 : 40.0
# Overboard: 2
# min_ 20
# max_:  80
# div_: 20
# cal_abs 절대값 : 17
# compare7 : 13.833333333333334
# compare6 : 16.6
# compare5 : 20.75
# compare4 : 27.666666666666668
# compare3 : 41.5
# Lacking: 2
# 최종결과 :  [1, 0, 2, 0, 2.75]


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
from pandas import DataFrame

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F
import time

nltk.download('averaged_perceptron_tagger')


def character_all_section(text):

    # Number of Characters
    def NumberofCharacters(text):

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
        ##print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
        #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
        filtered_chr_text = []
        for k in token_input_text:
            for j in character_list:
                if k == j:
                    filtered_chr_text.append(j)
        
        ##print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
        
        filtered_chr_text_ = set(filtered_chr_text) #중복제거
        filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
        #print (filtered_chr_text__) # 중복값 제거 확인
        
        # for i in filtered_chr_text__:
        #     ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
        char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 캐릭터 표현 수
        char_count_ = len(filtered_chr_text__) #중복제거된 캐릭터 표현 총 수
            
        result_char_ratio = round(char_total_count/total_words * 100, 2)
        return char_total_count



    # number_of_characters = NumberofCharacters(input_text) # 문장에서 키워드와 관련된 단어을 모두 추출하면 이런 결과가 나옴, 이 결과를 모두 합쳐서 캐릭터 총 값 계산해서 숫자로 출력
    # number_of_characters
    # #print ('=============================================')
    # #print ('Number of Characters :', number_of_characters)
    # #print ('=============================================')

    ####################################
    #### Character Descriptiveness #####
    ####################################

    def character_descrip(text):

        input_sentence = text

        def findSentence(input_sentence):
            result = []

            data = str(input_sentence)
            #data = input_sentence.splitlines()
            
            findText = ['i', 'my', 'me', 'mine', 'you', 'your', 'they','them',
                        'yours', 'he','him','his' 'she','her','it','someone','their', 'myself', 'aunt',
                        'brother','cousin','daughter','father','grandchild','granddaughter','granddson','grandfather',
                        'grandmother','great-grandchild','husband','ex-husband','son-in-law', 'daughter-in-law','mother',
                        'niece','nephew','parents','sister','son','stepfather','stepmother','stepdaughter', 'stepson',
                        'twin','uncle','widow','widower','wife','ex-wife','aunt',
                        'baby', 'beget', 'brother', 'buddy', 'conserve', 'counterpart', 'cousin', 'daughter', 'duplicate', 'ex',
                        'father', 'forefather', 'founder', 'gemini', 'grandchild', 'granddaughter', 'grandfather', 'grandma', 'he', 'helium',
                        'husband', 'i', 'in', 'iodine', 'law', 'maine', 'match', 'mine', 'mother', 'nephew', 'niece', 'one', 'parent', 'person',
                        'rear', 'sister', 'son', 'stepdaughter', 'stepfather', 'stepmother', 'stepson', 'twin', 'uncle', 'widow', 'widower', 'wife']


            sentences = data.split(".")
            
            for sentence in sentences:
                for item in findText:
                    if item in sentence:
                        result.append(sentence)

            return result

        input_sent_included_character = findSentence(text) 
        input_sent_chr = set(input_sent_included_character) #중복값을 제거해보자
        input_sent_chr = '.'.join(input_sent_chr) #하나의 문자열로 합쳐야 원본 문장처럼 변환되고, 이것을 show/tell 분석코드에 넣게됨



        #입력된 전체 문장을 개별문장으로 분리하여 전처리 처리함
        def sentence_to_df(input_sentence):

            input_text_df = nltk.tokenize.sent_tokenize(input_sentence)
            test = []

            for i in range(0,len(input_text_df)):
                new_label = np.random.randint(0,2)  # 개별문장(input_text_df) 수만큼 0 또는 1 난수 생성
                data = [new_label, input_text_df[i]]
                test.append(data)

            ##print(test)
            dataf = pd.DataFrame(test, columns=['label', 'text'])
            ##print(dataf)
            return dataf


        class STDataset(Dataset):
            ''' Showing Telling Corpus Dataset '''
            def __init__(self, df):
                self.df = df

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                text = self.df.iloc[idx, 1]
                label = int(self.df.iloc[idx, 0])
                return text, label


        ###########입력받은 데이터 처리 실행하는 메소드 showtell_classfy() ###############
        #result_all.html에서 입력받을 text를 contents에 넣고 전처리 후 데이터프레임에 넣어줌
        def showtell_classfy(text):
            contents = str(text)
            preprossed_contents_df = sentence_to_df(contents)

            preprossed_contents_df.dropna(inplace=True)
            #전처리된 데이터를 확인(데이터프레임으로 가공됨)
            preprossed_contents_df__ = preprossed_contents_df.sample(frac=1, random_state=999)
            

            #파이토치에 입력하기 위해서 로딩...
            ST_test_dataset = STDataset(preprossed_contents_df__)
            test_loader = DataLoader(ST_test_dataset, batch_size=1, shuffle=True, num_workers=0)
            #로딩되는지 확인
            ST_test_dataset.__getitem__(1)

            #time.sleep(1)



            #check whether cuda is available
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            device = torch.device("cpu")  
            #device = torch.device("cuda")
            #tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
            model = BertForSequenceClassification.from_pretrained('bert-large-cased')
            model.to(device)



            # for text, label in test_loader :
            #     #print("text:",text)
            #     #print("label:",label)


            #저장된 모델을 불러온다.
            #J:\Django\EssayFit_Django\essayfitaiproject\essayfitapp\model.pt
            #time.sleep(1)
            #model = torch.load("model.pt", map_location=torch.device('cpu'))
            model = torch.load("./essayai/data/model.pt", map_location=torch.device('cpu'))
            # #print("model loadling~")
            model.eval()


            pred_loader = test_loader
            # #print("pred_loader:", pred_loader)
            total_loss = 0
            total_len = 0
            total_showing__ = 0
            total_telling__ = 0

            showing_conunter = [] #문장에 해당하는 SHOWING을 계산한다.
            
            # #print("check!")
            for text, label in pred_loader:
                # #print("text:",text)
                ##print("label:",label)
                encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text] #text to tokenize
                padded_list =  [e + [0] * (512-len(e)) for e in encoded_list] #padding
                sample = torch.tensor(padded_list) #torch tensor로 변환
                sample, label = sample.to(device), label.to(device) #tokenized text에 label을 넣어서 Device(gpu/cpu)에 넣기 위해 준비
                labels = torch.tensor(label) #레이블을 텐서로 변환
                #time.sleep(1)
                outputs = model(sample,labels=labels) #모델을 통해서 샘플텍스트와 레이블 입력데이터를 출력 output에 넣음
                #시간 딜레이를 주자
                #time.sleep(1)
                _, logits = outputs #outputs를 로짓에 넣음 이것을 softmax에 넣으면 0~1 사이로 결과가 출력됨
                
                pred = torch.argmax(F.softmax(logits), dim=1) #드디어 예측한다. argmax는 리스트(계산된 값)에서 가장 큰 값을 추출하여 pred에 넣는다. 0 ~1 사이의 값이 나올거임
                # #print('pred :', pred)
                # correct = pred.eq(labels) 
                showing__ = pred.eq(1) # 예측한 결과가 1과 같으면 showing이다   >> TRUE   SHOWING을 추출하려면 이것만 카운드하면 된다. 
                telling__ = pred.eq(0) # 예측한 결과가 0과 같으면 telling이다   >> FALSE
                
                ##print('showing : ', showing__)
                ##print('telling : ', telling__)
                
                
                showing_conunter.append(text)        
                #pred_ = round(float(pred))
                showing_conunter.append(pred)


            return showing_conunter 


        st_re = showtell_classfy(str(input_sent_chr)) # 캐릭터거 포함된 문장(전처리 완료된) 입력

        df = DataFrame(st_re)
        df_ = df[0::2] # 글만 추출
        df_label = df[1::2] # 레이블만 추출

        df_.reset_index(drop=True, inplace=True) #데이터를 합치기 위해서 초기화
        df_label.reset_index(drop=True, inplace=True)

        df_result = pd.concat([df_,df_label],axis=1) #합치기

        df_result.columns = ['sentence','show/tell']

        df_fin = df_result['show/tell'].value_counts(normalize=True)
        list(df_fin)
        showing_sentence_with_char = max(round(df_fin*100))

        # #print("===============================================================")
        # #print ('Character Descriptiveness : ', showing_sentence_with_char)
        # #print("===============================================================")

        return showing_sentence_with_char



    ################################################
    #############  Emphasis on YOU  ################
    ################################################
    def EmphasisOnYou(text):

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

        #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고
        character_list = ['i', 'I', 'my', 'me', 'mine', 'one']
        
        ####문장에 char_list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
        #우선 토큰화한다.
        retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
        token_input_text = retokenize.tokenize(essay_input_corpus)
        ##print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
        #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
        filtered_chr_text = []
        for k in token_input_text:
            for j in character_list:
                if k == j:
                    filtered_chr_text.append(j)
        
        ##print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
        
        filtered_chr_text_ = set(filtered_chr_text) #중복제거
        filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
        # #print (filtered_chr_text__) # 중복값 제거 확인
        
        # for i in filtered_chr_text__:
        #     ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
        char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 캐릭터 표현 수
        char_count_ = len(filtered_chr_text__) #중복제거된 캐릭터 표현 총 수
            
        result_char_ratio = round(char_total_count/total_words * 100, 2)
        return char_total_count


    # EmphasisOnYou_ = EmphasisOnYou(input_text) # 문장에서 키워드와 관련된 단어을 모두 추출하면 이런 결과가 나옴, 이 결과를 모두 합쳐서 캐릭터 총 값 계산해서 숫자로 출력
    # EmphasisOnYou_
    # #print ('=============================================')
    # #print ('Emphasis on You :', EmphasisOnYou_)
    # #print ('=============================================')



    #########################################
    ######### Emphasis on others  ###########
    #########################################
    def EmphasisOnOthers(text):

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

        #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고
        character_list = ['they','them','he','him','his' 'she','her','it','someone','their', 'myself', 'aunt',
                        'brother','cousin','daughter','father','grandchild','granddaughter','granddson','grandfather',
                        'grandmother','great-grandchild','husband','ex-husband','son-in-law', 'daughter-in-law','mother',
                        'niece','nephew','parents','sister','son','stepfather','stepmother','stepdaughter', 'stepson',
                        'twin','uncle','widow','widower','wife','ex-wife','aunt',
                        'baby', 'beget', 'brother', 'buddy', 'conserve', 'counterpart', 'cousin',
                        'daughter', 'duplicate', 'ex', 'father', 'forefather', 'founder', 'gemini',
                        'grandchild', 'granddaughter', 'grandfather', 'grandma', 'he', 'helium', 'husband',
                        'in', 'law', 'match', 'mother', 'nephew', 'niece', 'parent', 'person', 'rear',
                        'sister', 'son', 'stepdaughter', 'stepfather', 'stepmother', 'stepson', 'twin', 'uncle', 'widow',
                        'widower', 'wife']
        
        ####문장에 char_list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
        #우선 토큰화한다.
        retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
        token_input_text = retokenize.tokenize(essay_input_corpus)
        ##print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
        #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
        filtered_chr_text = []
        for k in token_input_text:
            for j in character_list:
                if k == j:
                    filtered_chr_text.append(j)
        
        ##print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
        
        filtered_chr_text_ = set(filtered_chr_text) #중복제거
        filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
        #print (filtered_chr_text__) # 중복값 제거 확인
        
        # for i in filtered_chr_text__:
        #     ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
        char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 캐릭터 표현 수
        char_count_ = len(filtered_chr_text__) #중복제거된 캐릭터 표현 총 수
            
        result_char_ratio = round(char_total_count/total_words * 100, 2)
    
    
        return char_total_count


    # EmphasisOnOthers_ = EmphasisOnOthers(input_text) # 문장에서 키워드와 관련된 단어을 모두 추출하면 이런 결과가 나옴, 이 결과를 모두 합쳐서 캐릭터 총 값 계산해서 숫자로 출력
    # EmphasisOnOthers_
    # #print ('=============================================')
    # #print ('Emphasis on Others :', EmphasisOnOthers_)
    # #print ('=============================================')


    character_descriptiveness = character_descrip(text)
    # #print("===============================================================")
    # #print ('Character Descriptiveness : ' , character_descriptiveness)
    # #print("===============================================================")


    number_of_characters = NumberofCharacters(text) 
    # #print ('=============================================')
    # #print ('Number of Characters :' , number_of_characters)
    # #print ('=============================================')


    EmphasisOnYou_ = EmphasisOnYou(text)
    # #print ('=============================================')
    # #print ('Emphasis on You :' , EmphasisOnYou_)
    # #print ('=============================================')


    EmphasisOnOthers_ = EmphasisOnOthers(text) 
    # #print ('=============================================')
    # #print ('Emphasis on Others :' , EmphasisOnOthers_)
    # #print ('=============================================')


    return character_descriptiveness, number_of_characters, EmphasisOnYou_, EmphasisOnOthers_




###################################################################
###################################################################



########## 실행 테스트 ##########

# 한명의 에세이 데이터 입력하여 계산
text= """A window into the soul.For most people, this would be the eyes. The eyes cannot lie; they often tell more about a person's emotions than their words. What distinguishes a fake smile from a genuine one? The eyes. What shows sadness? The eyes. What gives away a liar? The eyes.But are the eyes the only window into the soul?Recently, I began painting with watercolors. With watercolors, there is no turning back: if one section is too dark, it is nearly impossible to lighten the area again. Every stroke must be done purposefully, every color mixed to its exact value.I laid my materials before me, preparing myself for the worst. I checked my list of supplies, making sure my setup was perfect.I wet my brush, dipped it into some yellow ochre, and dabbed off the excess paint. Too little water on my brush. I dipped my brush back into my trusty water jar; the colors swirled beautifully, forming an abstract art piece before my eyes. \u2014It's a shame that I couldn't appreciate it.I continued mixing colors to their exact value. More alizarin crimson. More water. More yellow ochre. Less water. More phthalo blue. The cycle continued. Eventually, I was satisfied. The colors looked good, there was enough contrast between facial features, and the watercolors stayed inside the lines.Craving feedback, I posted my art to Snapchat. I got a few messages such as 'wow' and 'pretty,' but one message stood out. 'You were anxious with this one, huh? Anyways, love the hair!'I was caught off guard. Was it a lucky guess? Did they know something I didn't? I immediately responded: 'Haha, how could you tell?' No response.What I didn't know at the time was that my response would come a few months later while babysitting. Since the girl I was babysitting loved art, I took out some Crayola watercolors and some watercolor paper for her to play with. After I went to the bathroom and came back, the watercolors were doused with water. 'You were impatient with this one, huh? Anyways, love the little dog you drew!'The little girl looked up at me, confused. 'How could you tell?' 'You used a lot of water for a brighter color, but you couldn't wait for it to slowly soak in.''Oh.'Now, I would be lying if I said I realized the connection between the two events immediately.Instead, I made the connection when I decided to sit down one day and objectively critique my art. The piece that I once loved now seemed like a nervous wreck: the paper was overworked, the brushstrokes were undecided, the facial features blended together, and each drop of water was bound inside the lines as if it was a prisoner in a cage.From then on, I started noticing pieces of personality in additional creations surrounding me: website designs, solutions to math problems, code written for class, and even the preparation of a meal.When I peer around at people's projects during Code Club, I notice the clear differences between their code. Some people break it up by commenting in every possible section. Others breeze through the project, not caring to comment or organize their code. I could also see clear differences in personalities when our club members began coding the Arduino for the first time. Some followed the tutorials to the letter, while others immediately started experimenting with different colored LEDs and ways of wiring the circuit.It became clear to me that, as humans, we leave pieces of our souls in everything we do, more than we intend to. If we entertain this thought, perhaps the key to better understanding others around us is simply noticing the subtler clues under our noses?Perhaps there are endless windows to the soul, and we simply need to peer through them. I shakily rose my hand. 'We should create workshops of our own,' I suggested.I got a few strange looks. 'It's a good idea, but it's too much work.' 'We just don't have enough free time to make it work.' 'Maybe we could, but I don't know how to make workshops.' My suggestion was shot down. I shuffled in my seat. 'I could make them.' A few people stared at me in disbelief. I glanced over at the club advisor, Mr. C, nervous to hear his response.'If you're willing to take on the work, we can try it.' Mr. C replied. And so I embarked on my quest. I researched different workshops on the internet, learning the information myself at first. Then, I transitioned into creating workshops of my own, making sure that the information was easy to understand for even a beginner. I was exhausted; my first workshop took 16 cumulative hours to create."""
# 

def character_total_analysis(text):

    ##### 주석 처리 ##########################
    ########################################
    char_sec_re = character_all_section(text)
    #char_sec_re = (67.0, 145, 66, 33)   ### dummy Data 

    # #print("1명의 에세이 결과 계산점수 :", char_sec_re)

    # 위에서 계산한 총 4개의 값을 개인, 그룹의 값과 비교하여 lacking, ideal, overboard 계산
    
    # 개인에세이 값 계산 4가지 결과 추출 >>>>> personal_value 로 입력됨
    one_ps_char_desc = char_sec_re[0] # 튜플에서 첫번재 인댁스 값 가져오기 : Character Descriptiveness
    one_ps_num_of_char = char_sec_re[1]
    one_ps_emp_on_you = char_sec_re[2]
    one_ps_emp_on_others = char_sec_re[3]

    ##############################################################################################################################
    ## 1000명 데이터의 각 값(char_desc_mean)의 평균 값 전달. >>>> group_mean 으로 입력됨
    char_desc_mean = [77, 478, 14, 50] # 현재 계산 완료한 1000명의 평균 값(고정값) 
    group_db_fin_result = [5.0] #레이다차트의 1000명 평균값 기준설정
    ##############################################################################################################################


    char_desc_ideal_mean = char_desc_mean[0] #첫번째 값을 가져옴, Character Descriptiveness
    num_of_char_ideal_mean = char_desc_mean[1] #Number of Characters
    emp_on_you_ideal_mean = char_desc_mean[2] #Emphasis on You 
    emp_on_others_ideal_mean = char_desc_mean[3] #Emphasis on Others


    def lackigIdealOverboard(group_mean, personal_value): # group_mean: 1000명 평균, personal_value|:개인값
        ideal_mean = group_mean
        one_ps_char_desc = personal_value
        #최대, 최소값 기준으로 구간설정. 구간비율 30% => 0.3으로 설정
        min_ = int(ideal_mean-ideal_mean*0.6)
        # #print('min_', min_)
        max_ = int(ideal_mean+ideal_mean*0.6)
        # #print('max_: ', max_)
        div_ = int(((ideal_mean+ideal_mean*0.6)-(ideal_mean-ideal_mean*0.6))/3)
        # #print('div_:', div_)

        #결과 판단 Lacking, Ideal, Overboard
        cal_abs = abs(ideal_mean - one_ps_char_desc) # 개인 - 단체 값의 절대값계산

        # #print('cal_abs 절대값 :', cal_abs)
        compare7 = (one_ps_char_desc + ideal_mean)/6
        compare6 = (one_ps_char_desc + ideal_mean)/5
        compare5 = (one_ps_char_desc + ideal_mean)/4
        compare4 = (one_ps_char_desc + ideal_mean)/3
        compare3 = (one_ps_char_desc + ideal_mean)/2
        # #print('compare7 :', compare7)
        # #print('compare6 :', compare6)
        # #print('compare5 :', compare5)
        # #print('compare4 :', compare4)
        # #print('compare3 :', compare3)



        if one_ps_char_desc > ideal_mean: # 개인점수가 평균보다 클 경우는 overboard
            if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
                # #print("Overboard: 2")
                result = 2 #overboard
                score = 1
            elif cal_abs > compare4: # 28
                # #print("Overvoard: 2")
                result = 2
                score = 2
            elif cal_abs > compare5: # 22
                # #print("Overvoard: 2")
                result = 2
                score = 3
            elif cal_abs > compare6: # 18
                # #print("Overvoard: 2")
                result = 2
                score = 4
            else:
                # #print("Ideal: 1")
                result = 1
                score = 5
        elif one_ps_char_desc < ideal_mean: # 개인점수가 평균보다 작을 경우 lacking
            if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
                # #print("Lacking: 2")
                result = 0
                score = 1
            elif cal_abs > compare4: # 28
                # #print("Lacking: 2")
                result = 0
                score = 2
            elif cal_abs > compare5: # 22
                # #print("Lacking: 2")
                result = 0
                score = 3
            elif cal_abs > compare6: # 18
                # #print("Lacking: 2")
                result = 0
                score = 4
            else:
                # #print("Ideal: 1")
                result = 1
                score = 5
                
        else:
            # #print("Ideal: 1")
            result = 1
            score = 5

        return result, score


    #종합계산시작 lackigIdealOverboard(group_mean, personal_value)
    character_desc_result = lackigIdealOverboard(char_desc_ideal_mean, one_ps_char_desc)
    number_of_char_result = lackigIdealOverboard(num_of_char_ideal_mean, one_ps_num_of_char)
    emp_on_you_result = lackigIdealOverboard(emp_on_you_ideal_mean, one_ps_emp_on_you)
    emp_on_others_result = lackigIdealOverboard(emp_on_others_ideal_mean, one_ps_emp_on_others)

    fin_result = [character_desc_result, number_of_char_result, emp_on_you_result, emp_on_others_result]

    # each_fin_result의 결과는 순서대로 [Character Descriptiveness, Number of Characters, Emphasis on You, Emphasis on Others] 이다. 0: lacking, 1:ideal, 2:overbaord 
    each_fin_result = [fin_result[0][0], fin_result[1][0], fin_result[2][0], fin_result[3][0]]
    # 최종 character  전체 점수 계산
    overall_character_rating = [(fin_result[0][1]+ fin_result[1][1] + fin_result[2][1]+ fin_result[3][1])/4]

    result_final = each_fin_result + overall_character_rating + group_db_fin_result
    
    data = {
        "number_of_chracters":one_ps_char_desc, 
        "character_description":one_ps_num_of_char,
        "emaphasis_on_you":one_ps_emp_on_you,
        "emaphasis_on_others":one_ps_emp_on_others,
        
        "result_number_of_chracters": result_final[0],
        "result_character_description": result_final[1],
        "result_emaphasis_on_you" : result_final[2],
        "result_emaphasis_on_others" : result_final[3],
        
        "avg_character": result_final[4]
    }
    

    return data 



#################### 테스트~~~~!!!! ###################
# character_total_analysis(text)

# print("최종결과 : ", character_total_analysis(text))

## 최종결과 :  {'number_of_chracters': 67.0, 'character_description': 145, 'emaphasis_on_you': 66, 
#             'emaphasis_on_others': 33, 'result_number_of_chracters': 1, 'result_character_description': 0, 
#             'result_emaphasis_on_you': 2, 'result_emaphasis_on_others': 0, 'avg_character': 2.75}

# 최종결과 :  [1, 0, 2, 0, 2.75, 5.0]
# 
# 결과 설명 : ideal, ideal, lacking, lacking, overall_character_rating, group_db_fin_result

# Character Descriptiveness: ideal
# Number of Characters : ideal
# Emphasis on You : lacking
# Emphasis on Others : lacking
# overall_character_rating : 2.75
# group_db_fin_result : 5.0  


## 실제 출력결과(확인용)

# ===============================================================
# Character Descriptiveness :  67.0
# ===============================================================
# ['person', 'i', 'them', 'her', 'you', 'they', 'me', 'one', 'myself', 'their', 'in', 'it', 'my']
# ai_character_section.py:128: DeprecationWarning: Call to deprecated `most_similar_cosmul` (Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead).
#   ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
# =============================================
# Number of Characters : 145
# =============================================
# ['i', 'me', 'one', 'my']
# ai_character_section.py:379: DeprecationWarning: Call to deprecated `most_similar_cosmul` (Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead).
#   ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
# =============================================
# Emphasis on You : 66
# =============================================
# ['person', 'them', 'her', 'they', 'myself', 'their', 'in', 'it']
# ai_character_section.py:455: DeprecationWarning: Call to deprecated `most_similar_cosmul` (Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead).
#   ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
# =============================================
# Emphasis on Others : 33
# =============================================
# 1명의 에세이 결과 계산점수 : (67.0, 145, 66, 33)
# min_ 30
# max_:  123
# div_: 30
# cal_abs 절대값 : 10.0
# compare7 : 24.0
# compare6 : 28.8
# compare5 : 36.0
# compare4 : 48.0
# compare3 : 72.0
# Ideal: 1
# min_ 191
# max_:  764
# div_: 191
# cal_abs 절대값 : 333
# compare7 : 103.83333333333333
# compare6 : 124.6
# compare5 : 155.75
# compare4 : 207.66666666666666
# compare3 : 311.5
# Lacking: 2
# min_ 5
# max_:  22
# div_: 5
# cal_abs 절대값 : 52
# compare7 : 13.333333333333334
# compare6 : 16.0
# compare5 : 20.0
# compare4 : 26.666666666666668
# compare3 : 40.0
# Overboard: 2
# min_ 20
# max_:  80
# div_: 20
# cal_abs 절대값 : 17
# compare7 : 13.833333333333334
# compare6 : 16.6
# compare5 : 20.75
# compare4 : 27.666666666666668
# compare3 : 41.5
# Lacking: 2
# 최종결과 :  [1, 0, 2, 0, 2.75, 5.0]