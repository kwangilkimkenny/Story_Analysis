#본 코드는 6개의 질문중 택 1의 경우 한명의 학생데이터와 1000명의 학생 데이터를 비교하여 상, 중, 하를 구분하는 코드임. 

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
from collections import defaultdict

nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from pandas import DataFrame as df
from mpld3 import plugins, fig_to_html, save_html, fig_to_dict
from tqdm import tqdm
import numpy as np
import json
from tensorflow.keras.preprocessing.text import text_to_word_sequence


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

# synonym: 동의어
# antonym: 반의어
# hypernym: 상의어
# hyponym: 하위어

# 여기서는 synonym 만 추출하여 추가 분석에 반영함
from nltk.corpus import wordnet as wn

#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import pandas as pd
import sys
import pickle


##### 질문을 선택하고, 에세이를 입력한다. #####

# 1명 학생의 입력데이터
input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""

# 6개의 질문  ques_one, ques_two, ques_three, ques_four, ques_five, ques_six   중 선택 1개
question_num = """ques_one""" # 1번째 질문을 선택했을 경우





### START  ###
def theme_all_section(input_text, question_num):


        # with open('personal_statement_980_fin.json','r') as json_file :
        #     json_data = json.load(json_file)



        # # 1000명의 학생 데이터를 추출
        # st_data_txt = json.dumps(json_data)

        # #데이터 확인완료
        #print(st_data_txt)




        ######################
        ##### QUESTION 1 #####
        ######################
        #표현하는 단어들을 리스트에 넣어서 필터로 만들고
        qst_one_words_list = ['identity', 'background', 'interest', 'talent', 'meaningful','belief', 'explore', 'develop',
                            'realize', 'unique', 'passion', 'different', 'culture', 'sex', 'gender', 'religion', 
                            'profession', 'major', 'ethnic', 'disability', 'excel', 'standout', 'diversity',
                            'acculturation','alone','arouse','backdrop','background','belief','break','concern','cultural',
                            'culture','develop','different','disability','diverseness','diversity','endowment','ethnic','evolve',
                            'excel','explicate','explore','gain','gender','grow','heat','heathen','identity','impression','interest',
                            'love','major','mania','meaningful','modernize','originate','passion','pastime','polish','profession',
                            'rage','realize','recognize','religion','research','sake','setting','sex','singular','talent','train',
                            'understand','unique','unlike']   



        ######################
        ##### QUESTION 2 #####
        ######################
        #표현하는 단어들을 리스트에 넣어서 필터로 만들고, WORDNET에서 유사단어 추출하여 적용완료!
        qst_two_words_list = ['obstacle', 'challenge', 'setback', 'failure', 'difficulty', 'despair', 'defeat', 'hindrance', 'impediment', 
                                'misfortune', 'trouble', 'handicap', 'stumble', 'hurdle','bankruptcy',
                                'challenge', 'defeat', 'despair', 'difficulty', 'disability', 'disable', 'disturb', 'failure', 'frustration',
                                'fuss', 'handicap', 'hindrance', 'hurdle', 'kill', 'lurch', 'misfortune', 'obstacle', 'obstruction', 'perturb',
                                'reverse', 'stumble', 'trip', 'trouble', 'vault', 'worry']



        ######################
        ##### QUESTION 3 #####
        ######################
        #표현하는 단어들을 리스트에 넣어서 필터로 만들고, WORDNET에서 유사단어 추출하여 적용완료!
        qst_tree_words_list = ['idea', 'belief', 'question', 'thinking', 'prompted', 'outcome', 'challenge', 'defy', 'realize', 
                                'enlighten', 'philosophy', 'religion', 'conviction', 'believe', 'thoughts', 'reason', 'logic', 'value', 
                                'conscience', 'ethic', 'right', 'justice', 'dare', 'concept', 'existing', 'inspire', 'confront', 'oppose', 
                                'conflict', 'against','argue', 'battle', 'belief', 'believe', 'cause', 'challenge', 'cheer', 'clear', 'concept', 'conflict',
                                'confront', 'conscience', 'consequence', 'conviction', 'correct', 'correctly', 'dare', 'defy', 'dispute', 'doctrine',
                                'doubt', 'enlighten', 'estimate', 'ethic', 'exist', 'existent', 'existing', 'fight', 'gain', 'good', 'idea', 'impression',
                                'inhale', 'inspire', 'intelligent', 'intend', 'interrogate', 'interview', 'judge', 'justice', 'justly', 'logic', 'measure',
                                'mighty', 'mind', 'motion','motivate', 'opinion', 'oppose', 'philosophy', 'pit', 'prize', 'prompt', 'proper', 'properly',
                                'question', 'r', 'rate', 'rationality', 'react', 'realize', 'reason', 'recognize', 'religion', 'remember', 'respect', 'result',
                                'revolutionize', 'right', 'theme', 'think', 'thinking', 'thought', 'understand', 'value', 'veracious', 'wonder' ]



        ######################
        ##### QUESTION 4 #####
        ######################
        #표현하는 단어들을 리스트에 넣어서 필터로 만들고, WORDNET에서 유사단어 추출하여 적용완료!
        qst_four_words_list = ['problem', 'solve', 'problem-solving', 'problem solving', 'intellectual', 'research','ethical dilemma', 'personal',
                                'significance', 'solution', 'identify','challenge', 'question', 'dilemma', 'dispute', 'answer', 'clarify', 
                                'figure out', 'work out', 'fix', 'conclude', 'realize', 'discover','answer',
                                'cerebral', 'challenge', 'clarify', 'clear', 'conclude', 'cook', 'detect', 'dilemma', 'discover', 'dispute',
                                'doubt', 'fasten', 'fix', 'fixate', 'gain', 'identify', 'inquiry', 'intellectual', 'interrogate', 'interview',
                                'learn', 'localization', 'meaning', 'motion', 'name', 'personal', 'problem', 'quarrel', 'question', 'realize',
                                'reason', 'recognize', 'repair', 'research', 'resolve', 'significance', 'situate', 'solution', 'solve', 'specify',
                                'sterilize', 'suffice', 'trouble', 'understand', 'unwrap', 'wonder']



        ######################
        ##### QUESTION 5 #####
        ######################
        #표현하는 단어들을 리스트에 넣어서 필터로 만들고, WORDNET에서 유사단어 추출하여 적용완료!
        qst_five_words_list = ['accomplishment', 'event', 'realization', 'spark', 'growth', 'understanding', 'myself', 'others'
                                'realization', 'realize', 'accomplish', 'event', 'incident', 'happening', 'understanding', 'insight', 'insightful', 'mature', 'maturity',
                                'growth', 'enlightenment', 'enlighten', 'perspective', 'empathize', 'empathy', 'sympathize', 'sympathy', 'appreciate',
                                'acknowledge', 'respect', 'humble','accomplishment', 'achieve', 'acknowledge', 'admit', 'adulthood', 'agreement', 'appreciate',
                                'base', 'clear', 'commiserate', 'consequence', 'deference', 'discharge', 'emergence', 'empathy', 'enlighten', 'enlightenment', 'esteem',
                                'event', 'find', 'fledged', 'flicker', 'gain', 'growth', 'happen', 'happening', 'humble', 'humiliate', 'incident', 'incidental', 'increase',
                                'insight', 'insightful', 'mature', 'maturity', 'nirvana', 'notice', 'obedience', 'penetration', 'perspective', 'position', 'prize', 'realization',
                                'realize', 'reason', 'recognize', 'regard', 'respect', 'ripe', 'ripen', 'senesce', 'skill', 'spark', 'sparkle', 'suppurate', 'sympathize',
                                'sympathy', 'trip', 'understand', 'understanding']



        ######################
        ##### QUESTION 6 #####
        ######################
        #표현하는 단어들을 리스트에 넣어서 필터로 만들고, WORDNET에서 유사단어 추출하여 적용완료!
        qst_six_words_list = ['topic', 'idea', 'concept', 'engaging', 'captivate', 'learn'
                            'learn', 'research', 'subject', 'mentor', 'teacher', 'professor', 'inspiration', 'study', 'fascinate', 'engross', 'discover', 'find',
                            'theory', 'thought', 'think', 'mesmerize', 'delve', 'inquiry', 'inquire', 'question', 'inquisitive', 'investigate',
                            'explore', 'absorb', 'analyze', 'ask', 'betroth', 'capable', 'capture', 'cogitation', 'concept', 'detect', 'determine',
                            'dig', 'discipline','discover', 'discovery', 'doubt', 'engage', 'engaging', 'estimate', 'explore', 'fascinate', 'find', 'hire', 'hypnotize',
                            'hypothesis', 'idea', 'identify', 'inhalation', 'inquiry', 'inquisitive', 'inspiration', 'intend', 'interrogate', 'interview', 'intrigue',
                            'investigate', 'learn', 'lease', 'magnetize', 'mentor', 'mind', 'motion', 'national', 'opinion', 'professor', 'prosecute', 'question',
                            'receive', 'recover', 'remember', 'report', 'research', 'rule', 'sketch', 'steep', 'study', 'subject', 'subjugate', 'submit', 'survey',
                            'teacher', 'theme', 'theory', 'think', 'thinking', 'thought', 'topic', 'unwrap', 'witness', 'wonder']




        ####  문항을 선택하고 에세이를 입력했을 경우. 선택문항관련 단어리스트와 입력한 에세이의 공통적인 연관어 추출  ####


        #text : 입력 에세이
        #question_num_list : 선택한 질문과 연관된 단어 리스트

        def sim_words_quesiton(text_input, question_num_list):

            essay_input_corpus = str(text_input) #문장입력
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

            ######################
            ##### QUESTION 1 ~6 ## 관련 단어는 다른 함수에서 처리하여 적용할 것!!  문항 1~6번을 선택했을 경우 이하 코드 계산(이것은 클래스로 선언)
            ######################
                
            ####문장에 list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.

            #우선 토큰화한다.
            retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
            token_input_text = retokenize.tokenize(essay_input_corpus)
            #print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
            #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
            filtered_chr_text = []
            for k in token_input_text:
                for j in question_num_list:
                    if k == j:
                        filtered_chr_text.append(j)

            #print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.

            filtered_chr_text_ = set(filtered_chr_text) #중복제거
            filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
            #print (filtered_chr_text__) # 중복값 제거 확인

            # for i in filtered_chr_text__:
            #     ext_sim_words_key = model.most_similar_cosmul(i,topn=50) #모델적용

            # char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 표현 수
            # char_count_ = len(filtered_chr_text__) #중복제거된  표현 총 수

            # result_char_ratio = round(char_total_count/total_words * 100, 2)

            # import pandas as pd

            # df_conf_words = pd.DataFrame(ext_sim_words_key, columns=['words','values']) #데이터프레임으로 변환
            # df_r = df_conf_words['words'] #words 컬럼 값 추출
            # ext_sim_words_key = df_r.values.tolist() # 유사단어 추출


            ext_sim_words_key = filtered_chr_text_ 
            #return result_char_ratio, total_sentences, total_words, char_total_count, char_count_, ext_sim_words_key
            return ext_sim_words_key



        ########## 선택한 질문에 의해 해당하는 코드가 실행되는 부분  ###########

        if 'ques_one' == question_num: #선택한 질문이 ques_one 이면

            result_ques_ = sim_words_quesiton(input_text, qst_one_words_list) #입력한 에세이에 관하여 관련단어를 추출을 시작하라
            print("질문 1에 해당하는 1명 데이터 관련어 :", result_ques_)
            
            # load
            with open('question_one_1000_dataset.pickle', 'rb') as f:
                result_most_simWords = pickle.load(f)
            print("1000명 관련 data loaded :", result_most_simWords)

        elif 'ques_two' == question_num:

            result_ques_ = sim_words_quesiton(input_text, qst_two_words_list)
            print("result_ques_two :", result_ques_)
            
            # load
            with open('question_two_1000_dataset.pickle', 'rb') as f:
                result_most_simWords = pickle.load(f)
            print("1000명 관련 data loaded :", result_most_simWords)

        elif 'ques_three' == question_num:

            result_ques_ = sim_words_quesiton(input_text, qst_three_words_list)
            print("result_ques_three :", result_ques_)
            
            # load
            with open('question_three_1000_dataset.pickle', 'rb') as f:
                result_most_simWords = pickle.load(f)
            print("1000명 관련 data loaded :", result_most_simWords)


        elif 'ques_four' == question_num:

            result_ques_ = sim_words_quesiton(input_text, qst_four_words_list)
            print("result_ques_four :", result_ques_)
            
            # load
            with open('question_four_1000_dataset.pickle', 'rb') as f:
                result_most_simWords = pickle.load(f)
            print("1000명 관련 data loaded :", result_most_simWords)
            
        elif 'ques_five' == question_num:

            result_ques_ = sim_words_quesiton(input_text, qst_five_words_list)
            print("result_ques_five :", result_ques_)
            
            # load
            with open('question_five_1000_dataset.pickle', 'rb') as f:
                result_most_simWords = pickle.load(f)
            print("1000명 관련 data loaded :", result_most_simWords)
            
        elif 'ques_six' == question_num:

            result_ques_ = sim_words_quesiton(input_text, qst_six_words_list)
            print("result_ques_six :", result_ques_)
            
            # load
            with open('question_six_1000_dataset.pickle', 'rb') as f:
                result_most_simWords = pickle.load(f)
            print("1000명 관련 data loaded :", result_most_simWords)    

        else:
            print("let me think...")
            pass


        #############################################################################################################

        ##### 1번 문항에 해당하는 1000명의 학생 에세이 분석결과 ####
        # 유사단어를 문장에서 추출하여 반환한다.
        # st_data_txt >> 1000명의 에세이이다.
        # qst_one_words_list >>> 1번재 질문을 선택했을 경우다.

        #que_no_one_sim_words_ratio_result = sim_words_quesiton(st_data_txt, qst_one_words_list)

        # 위 결과(학생에세이분석결과)를 하나씩 꺼내서 1000명에세이 분석결과와 Doc2Vec로 개별 비교한다.
        # 분석하기 위하여 입력데이터 전처리  예 ) ['학생데이터리스트중 1개', '나머지는 1000명의 데이터리스트']
        # 6번 처리해야 하리때문에 함수로 변환적용할것!!!!
        import numpy as np

        #분석데이터 합치기
        input_data_preprocessed = []
        for std_keyword in result_ques_:# 위 결과(학생에세이분석결과)를 하나씩 꺼내서 
            input_data_preprocessed.append(std_keyword) #리스트에 첫 단어를 담고, 나머지 리스트데이터는 1000명것을 붙여넣는다.
            for item_ in result_most_simWords:
                input_data_preprocessed.append(item_) #리스트 합치기
            #input_data_preprocessed.append('.') #리스트를 구분한다. '.'로 구분

        #input_data_preprocessed #분석데이터 합친 결과 리스트, 이 리스트 데이터를 구간별(학생1단어, 1000개 단어가 1set)로 나누어서  DOC2VEC를 적용해보자 한번에 싹 처리해부러~

        #질문에 대한 1명의 학생에세이 분석결과
        ps_documents_df=pd.DataFrame(result_ques_, columns=['documents_cleaned'])


        def most_similar(doc_id,similarity_matrix, matrix):
            print (f'대표 WORD: {ps_documents_df.iloc[doc_id]["documents_cleaned"]}')
            print ('\n')
            print (f'Similar Words using {matrix}:')
            if matrix=='Cosine Similarity':
                similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
            elif matrix=='Euclidean Distance':
                similar_ix=np.argsort(similarity_matrix[doc_id])
                
            re_simil_words = []
            re_simil_cos = []
            for ix in similar_ix:
                if ix==doc_id:
                    continue
                #print('\n')
                #print (f'{ps_documents_df.iloc[ix]["documents_cleaned"]} {similarity_matrix[doc_id][ix]}')
                re_simil_words.append(ps_documents_df.iloc[ix]["documents_cleaned"])
                re_simil_words.append(similarity_matrix[doc_id][ix])
        #        print (f'{matrix} : {similarity_matrix[doc_id][ix]}')
        #         print (f'Word: {ps_documents_df.iloc[ix]["documents_cleaned"]}')
        #         print (f'{matrix} : {similarity_matrix[doc_id][ix]}')
            return re_simil_words,re_simil_cos



        #####  이걸 실행하라고~!
        def doctovec_run(input_value):
            #1번 질문에 대한 1명의 학생에세이 분석결과
            ps_documents_df=pd.DataFrame(input_value, columns=['documents_cleaned'])
            tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(ps_documents_df.documents_cleaned)]
            
            model_d2v = Doc2Vec(vector_size=100,alpha=0.0025, min_count=1)

            model_d2v.build_vocab(tagged_data)

            for epoch in range(100):
                model_d2v.train(tagged_data,
                            total_examples=model_d2v.corpus_count,
                            epochs=model_d2v.epochs)
                
            document_embeddings=np.zeros((ps_documents_df.shape[0],100))

            for i in range(len(document_embeddings)):
                document_embeddings[i]=model_d2v.docvecs[i]
                
            pairwise_similarities=cosine_similarity(document_embeddings)
            

            re_most_simWords = most_similar(0,pairwise_similarities,'Cosine Similarity',ps_documents_df)
            
            print("re_most_sim_words :" , re_most_simWords)
            
            return re_most_simWords

            ##################################################################################### 

            # def get_quet_numb_len(std_essay_dataset, selected_que_words):
            
            #     #1000명의 에세이, 선택된 질문관련 단어입력
                
            #     que_no_one_sim_words_ratio_result = sim_words_quesiton(std_essay_dataset, selected_que_words)
            #     quet_numb_len = len(que_no_one_sim_words_ratio_result)
                
            #     return quet_numb_len

            ############################################

        import time
        from tqdm import tqdm

        result_most_simWords = []

        cont = 0
        #이 코드는 문제없이 잘 돌아감
        for j in tqdm(result_ques_): #학생데이터를 하나씩 가져와서
            for k in tqdm(input_data_preprocessed): # 합친데이터를 하나씩 꺼내서
                if j == k: #같으면, 그 위치로부터 시작해서 비교 구간까지의 데이터를 꺼내온다.
                    print('j',j)
                    print('k',k)

                    input_data_preprocessed  #1명과 1000명데이터 분석결과 합친결과(1단어:1000명단어)

                    input_data_preprocessed_length = len(input_data_preprocessed) #1000명 처리 결과 길이구하기


                    end_numb = input_data_preprocessed.index(j) + len(result_most_simWords) + 1
                    print("input_data_preprocessed.index(j) : ", input_data_preprocessed.index(j))
                    print("end_numb :", end_numb)
                    in_text = input_data_preprocessed[input_data_preprocessed.index(j):end_numb]
                    print('분석할 단어 그룹', in_text)
                    # 첫 계산(학생 키워드와 전체 키워드 데이터의 거리를 각각 계산)을 하고, 다음 구간으로 넘어가자

                    #doctovec_run(in_text) #함수실행

                    #1번 질문에 대한 1명의 학생에세이 분석결과
                    ps_documents_df=pd.DataFrame(in_text, columns=['documents_cleaned'])
                    tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(ps_documents_df.documents_cleaned)]

                    model_d2v = Doc2Vec(vector_size=100,alpha=0.0025, min_count=1)

                    model_d2v.build_vocab(tagged_data)

                    for epoch in tqdm(range(100)):
                        model_d2v.train(tagged_data,
                                    total_examples=model_d2v.corpus_count,
                                    epochs=model_d2v.epochs)

                    document_embeddings=np.zeros((ps_documents_df.shape[0],100))

                    for i in range(len(document_embeddings)):
                        document_embeddings[i]=model_d2v.docvecs[i]


                    pairwise_similarities=cosine_similarity(document_embeddings)
                    #print('pairwise_similarities ::::::::::' , pairwise_similarities)

                    re_most_simWords = most_similar(cont, pairwise_similarities,'Cosine Similarity')
                    cont += 1
                    result_most_simWords.append(re_most_simWords)
                    print('re_most_simWords :', re_most_simWords)



        # 1명의 에세이와 선택한 문항과의 연관단어 : result_ques_, 연관단어와 1000명의 통계데이터 비교값   분산을 계산하자.

        return result_most_simWords



############# 실행 테스트 ###########

print("RESULT :", theme_all_section(input_text, question_num))
