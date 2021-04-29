import numpy as np
import spacy
from collections import Counter
import re
import nltk
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load("en_core_web_sm")


from wordcloud import WordCloud 
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
#%matplotlib inline

import matplotlib
from IPython.display import set_matplotlib_formats
matplotlib.rc('font',family = 'Malgun Gothic')
set_matplotlib_formats('retina')
matplotlib.rc('axes',unicode_minus = False)

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
from gensim import corpora, models, similarities

### sentence_simility.py ###
from sentence_similarity import sent_sim_analysis_with_bert_summarizer

### General Academic Knowledge ###
from general_academic_knowledge import GeneralAcademicKnowledge

### meaningfulExperienceLessonLearned.py
from meaningfulExperienceLessonLearned import MeaningFullExpreenceLessonLearened

### achievementYouAreProud.py
from achievementYouAreProud import get_achievement_you_are_proud_of

### social issue contribution 의 분석 중 topic uniqueness 분석
from topic_uniqueness import google_search_result

### social issue contribution 의 분석 중 topic knowledge 분석
from topic_knowledge import google_search_result_tp_knowledge

### social issue contribution 의 분석 중 2개의 결과 추출
from social_issue_contribution_solution import social_awareness_analysis
from social_issue_contribution_solution import initiative_engagement_contribution

### summer activity 
from summerActivity import SummerActivity
from summerActivity import summer_activity_initiative_engagement







#########################################################################
# Prompt Oriented Sentiments  -- 글속에 감정이 얼마나 표현되어 있는지 분석 - origin (Bert pre trained model 활용)
from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from pprint import pprint

tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
)

def Prompt_Oriented_Sentiments_analysis(essay_input):
    ########## 여기서는 최초 입력 에세이를 적용한다. input_text !!!!!!!!
    re_text = essay_input.split(".")

    #데이터 전처리 
    def cleaning(datas):

        fin_datas = []

        for data in datas:
            # 영문자 이외 문자는 공백으로 변환
            only_english = re.sub('[^a-zA-Z]', ' ', data)
        
            # 데이터를 리스트에 추가 
            fin_datas.append(only_english)

        return fin_datas

    texts = cleaning(re_text)

    #분석된 감정만 추출
    emo_re = goemotions(texts)

    emo_all = []
    for list_val in range(0, len(emo_re)):
        #print(emo_re[list_val]['labels'],emo_re[list_val]['scores'])
        #mo_all.append((emo_re[list_val]['labels'],emo_re[list_val]['scores'])) #KEY, VALUE만 추출하여 리스트로 저장
        #emo_all.append(emo_re[list_val]['scores'])
        emo_all.append((emo_re[list_val]['labels']))
        
    #추출결과 확인 
    # emo_all

    # ['sadness'],
    #  ['anger'],
    #  ['admiration', 'realization'],
    #  ['admiration', 'disappointment'],
    #  ['love'],
    #  ['sadness', 'neutral'],
    #  ['realization', 'neutral'],
    #  ['neutral'],
    #  ['optimism'],
    #  ['neutral'],
    #  ['excitement'],
    #  ['neutral'],
    #  ['neutral'],
    #  ['caring'],
    #  ['gratitude'],
    #  ['admiration', 'approval'], ...

    from pandas.core.common import flatten #이중리스틀 FLATTEN하게 변환
    flat_list = list(flatten(emo_all))

    # ['neutral',
    #  'neutral',
    #  'sadness',
    #  'anger',
    #  'admiration',
    #  'realization',
    #  'admiration',
    #  'disappointment',


    #중립적인 감정을 제외하고, 입력한 문장에서 다양한 감정을 모두 추출하고 어떤 감정이 있는지 계산해보자
    unique = []
    for r in flat_list:
        if r == 'neutral':
            pass
        else:
            unique.append(r)

    #중립감정 제거 및 유일한 감정값 확인
    #unique
    unique_re = set(unique) #중복제거

    ############################################################################
    # 글에 표현된 감정이 얼마나 다양한지 분석 결과!!!¶
    # print("====================================================================")
    # print("에세이에 표현된 다양한 감정 수:", len(unique_re))
    # print("====================================================================")

    #분석가능한 감정 총 감정 수 - Bert origin model 적용시 28개 감정 추출돰
    total_num_emotion_analyzed = 28

    # 감정기복 비율 계산 !!!
    result_emo_swings =round(len(unique_re)/total_num_emotion_analyzed *100,1) #소숫점 첫째자리만 표현
    # print("문장에 표현된 감정 비율 : ", result_emo_swings)
    # print("====================================================================")

    # 결과해서
    # reslult_emo_swings : 전체 문장에서의 감정 비율 계산
    # unique_re : 에세이에서 분석 추출한 감정   ====> 이것이 중요한 값임
    return result_emo_swings, unique_re



def select_prompt_type(prompt_type):
    
    if prompt_type == 'Why us':
        pmt_typ = [""" 'Why us' school & major interest (select major, by college & department) """]
        pmt_sentiment = ['Admiration', 'Excitement', 'Pride', 'Realization', 'Curiosity']
    elif prompt_type == 'Intellectual interest':
        pmt_typ = [""" Intellectual interest """]
        pmt_sentiment = ['Curiosity', 'Realization']
    elif prompt_type == 'Meaningful experience & lesson learned':
        pmt_typ = ["Meaningful experience & lesson learned"]
        pmt_sentiment = ['Realization', 'Approval', 'Gratitude', 'Admiration']
    elif prompt_type ==  'Achievement you are proud of':
        pmt_typ = ["Achievement you are proud of"]
        pmt_sentiment = ['Realization', 'Approval', 'Gratitude', 'Admiration', 'Pride', 'Desire', 'Optimism']
    elif prompt_type ==  'Social issues: contribution & solution':
        pmt_typ = ["Social issues: contribution & solution"]
        # 0~16 번째는 에세이 입력의 40% 분석 적용, 그 이후부분은 60% 적용  --> 즉 [:16], [17:] 이렇게 나눌 것
        pmt_sentiment = ['Anger', 'annoyance', 'Fear', 'Disapproval', 'disgust', 'Disappointment','grief', 'nervousness', 'sadness', 'surprise', 'remorse', 'curiosity', 'embarrassment', 'Realization','Approval', 'Gratitude', 'Admiration','Admiration','Approval', 'Caring', 'Joy', 'Gratitude', 'Optimism','relief', 'Realization']
    elif prompt_type ==  'Summer activity':
        pmt_typ = ["Summer activity"]
        pmt_sentiment = ['Pride','Realization','Curiosity','Excitement','Amusement','Caring']
    elif prompt_type ==  'Unique quality, passion, or talent':
        pmt_typ = [""]
        pmt_sentiment = ['Pride','Excitement','Amusement','Approval','Admiration','Curiosity']
    elif prompt_type ==  'Extracurricular activity or work experience':
        pmt_typ = [""]
        pmt_sentiment = ['Pride','Realization','Curiosity','Joy','Excitement','Amusement','Caring','Optimism']
    elif prompt_type ==  'Your community: role and contribution in your community':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Pride','Gratitude','Love']
    elif prompt_type ==  'College community: intended role, involvement, and contribution in college community':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Excitement','Pride','Gratitude']
    elif prompt_type ==  'Overcoming a Challenge or ethical dilemma':
        pmt_typ = [""]
        pmt_sentiment = ['Anger','Fear','Disapproval','Disappointment','Confusion','Annoyed','Realization', 'Approval','Gratitude','Admiration','Relief','Optimism']
    elif prompt_type ==  'Culture & diversity':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Realization','Love','Approval','Pride','Gratitude']
    elif prompt_type ==  'Collaboration & teamwork':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Optimism','Gratitude','Love']
    elif prompt_type ==  'Creativity/creative projects':
        pmt_typ = [""]
        pmt_sentiment = ['Excitement','Realization','Curiosity','Desire','Amusement','Surprise']
    elif prompt_type ==  'Leadership experience':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Optimism','Gratitude','Love','Fear','Confusion','Nervousness']
    elif prompt_type ==  'Values, perspectives, or beliefs':
        pmt_typ = [""]
        pmt_sentiment = ['Anger','Fear','Disapproval','Disappointment','Realization','Approval','Gratitude','Admiration']
    elif prompt_type ==  'Person who influenced you':
        pmt_typ = [""]
        pmt_sentiment = ['Realization', 'Approval', 'Gratitude','Admiration','Caring','Love','Curiosity', 'Pride', 'Joy']
    elif prompt_type ==  'Favorite book/movie/quote':
        pmt_typ = [""]
        pmt_sentiment = ['Excitement', 'Realization', 'Curiosity','Admiration','Amusement','Joy']
    elif prompt_type ==  'Write to future roommate':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Realization','Love','Excitement','Approval','Pride','Gratitude','Amusement','Curiosity','Joy']
    elif prompt_type ==  'Diversity & Inclusion Statement':
        pmt_typ = [""]
        pmt_sentiment = ['Anger','Fear','Disapproval','Disappointment','Confusion','Annoyed','Realization','Approval','Gratitude','Admiration','Relief','Optimism']
    elif prompt_type ==  'Future goals or reasons for learning':
        pmt_typ = [""]
        pmt_sentiment = ['Realization','Approval','Gratitude','Admiration','Pride','Desire','Optimism']
    elif prompt_type ==  'What you do for fun':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration', 'Excitement', 'Curiosity', 'Amusement', 'Pride','Joy']
    else:
        pass

    # pmt_typ : prompt type 
    # pmt_sentiment : prompy type에 해당하는 sentiment
    return pmt_typ, pmt_sentiment


#데이터 전처리 
def cleaning(data):
    fin_data = []
    for data_itm in data:
        # 영문자 이외 문자는 공백으로 변환
        only_english = re.sub('[^a-zA-Z]', ' ', data_itm)
        lists_re_ = re.sub(r"^\s+|\s+$", "", only_english) # 공백문자 제거
        only_english_ = lists_re_.rstrip('\n')
        # 데이터를 리스트에 추가 
        fin_data.append(only_english_)
    return fin_data


# txt 문서 정보 불러오기 : 대학정보
def open_data(select_college):
    # 폴더 구조, 대학이름 입력 명칭을 통일해야 함
    file_path = "./college_info/college_dataset/"
    college_name = select_college
    file_name = "_college_general_info.txt"
    # file = open("./college_info/colleges_dataset/brown_college_general_info.txt", 'r')
    file = open(file_path + college_name + file_name, 'r')
    lists = file.readlines()
    file.close()
    lists_re =  cleaning(lists) # 영어 단어가 아닌 것은 삭제
    result = ' '.join(lists_re) # 문장으로 합치기
    # 소문자 변환
    result_ = result.lower()
    #print("입력문장 불러오기 확인 : ", result_)
    return result_


# txt 문서 정보 불러오기 : 선택한 전공관련 정보 추출 
# 입력값은 대학, 전공 ex) 'Browon', 'AfricanStudies'
def open_major_data(select_college, select_major):
    # 폴더 구조, 대학이름 입력 명칭을 통일해야 함
    file_path = "./major_info/major_dataset/"
    college_name = select_college
    mjr_name = select_major
    file_name = "_major_info.txt"
    # file = open("./major_info/major_dataset/Brown_AfricanStudies_major_info.txt", 'r')
    file = open(file_path + college_name + "_" + mjr_name + file_name, 'r')
    lists = file.readlines()
    file.close()
    doc = ' '.join(lists)
    return lists


# 대학관련 정보의 토픽 키워드 추출하여 WordCloud로 구현
def general_keywords(College_text_data):
    tokenized = nltk.word_tokenize(str(College_text_data))
    #print('tokenized:', tokenized)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if(pos[:2] == 'NN')]
    count = Counter(nouns)
    words = dict(count.most_common())
    #print('words:', words)
    # 가장 많이 등장하는 단어를 추려보자. 
    wordcloud = WordCloud(background_color='white',colormap = "Accent_r",
                            width=1500, height=1000).generate_from_frequencies(words)

    plt.imshow(wordcloud)
    plt.axis('off')
    gk_re = plt.show()

    return gk_re






### 첫번째 방법: college_dept text data와 입력한 에세이의 데이터를 비교하여 TFD-IDF 유사도를 추출한다. ###
# 입력값 #
# College_dept_data : 대학별 수집된 '개별대학의 정보'입력
# Essay_input_data : 학생의 College Supp Essay 입력
# 이 코드는 완벽하게 일치하는 결과로 나오기때문에 사용하지 않고 새로운 코드를 적용해볼 것
def college_n_dept_fit(College_dept_data, Essay_input_data): # ----> 사용하지 말것, 하지만 백업코드

    documents = [College_dept_data]
    
    # remove common words and tokenize them
    stoplist = set('for a of the and to in'.split())

    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

    # remove words those appear only once
    all_tokens = sum(texts, [])

    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) ==1)
    texts = [[word for word in text if word not in tokens_once]
            for text in texts]
    dictionary = corpora.Dictionary(texts)

    dictionary.save('college_dept.dict')  # save as binary file at the dictionary at local directory
    dictionary.save_as_text('college_dept_text.dict')  # save as text file at the local directory

    #input college supp essay
    text_input = Essay_input_data #문장입력....

    new_vec = dictionary.doc2bow(text_input.lower().split()) # return "word-ID : Frequency of appearance""
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('college_dept.mm', corpus) # save corpus at local directory
    corpus = corpora.MmCorpus('college_dept.mm') # try to load the saved corpus from local
    dictionary = corpora.Dictionary.load('college_dept.dict') # try to load saved dic.from local
    tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
    corpus_tfidf = tfidf[corpus]  # map corpus object into tfidf space
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20) # initialize LSI #### num_topics=2
    corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus
    topic = lsi.print_topics(20)
    lsi.save('model.lsi')  # save output model at local directory
    lsi = models.LsiModel.load('model.lsi') # try to load above saved model

    doc = text_input

    vec_bow = dictionary.doc2bow(doc.lower().split())  # put newly obtained document to existing dictionary object
    vec_lsi = lsi[vec_bow] # convert new document (henceforth, call it "query") to LSI space
    index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and indexize it
    index.save('college_dept.index') # save index object at local directory
    index = similarities.MatrixSimilarity.load('college_dept.index')
    sims = index[vec_lsi] # calculate degree of similarity of the query to existing corpus

    # print(list(enumerate(sims))) # output (document_number , document similarity)

    sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sort output object as per similarity ( largest similarity document comes first )
    # print(sims) # 가장 유사도가 높은 순서대로 출력
    
    # result_sims = []
    
    sim_dic = {}
    
    for temp in sims : 
        
        sim_dic[temp[0]] = round(float(temp[1]),3) # 여기서 3은 top 3개의 일치도가 높은 순서대로 추출하라는 것, 여기서는 한개만 추출(한개 입력했으니)
        
        # result_sims.append([temp[0],round(float(temp[1]),3)])

    return sim_dic
    # 결과: 're_coll_n_dept_fit': {0: 1.0}}  로 입력한 두개의 문장 일치도가 1.0은 100% 같다는 의미다.


### 두 번째 방법 유사도 측정 ####




# Selected College 외 다양한 것을 계산하는 코드(최종계산코드)
# 입력값:  대학, 전공 ex) ('Why us', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
# 입력값:  대학, 전공 ex) ('Intellectual interest', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
def selected_college(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data):

    pmt_sent_etc_re = select_prompt_type(select_pmt_type)
    prompt_type_sentence = pmt_sent_etc_re[0] # prompt 문장 ex) Prompt Type : Intellectual interest 이렇게 20가지가 있음
    pmt_sent_re = list(pmt_sent_etc_re[1]) # prompt 에 해당하는 sentiment list
    intended_mjr = select_major # 희망전공

    if select_college == 'Harvard':
        pass
    elif select_college == 'Princeton':
        pass
    elif select_college == 'Stanford':
        pass
    elif select_college == 'MIT':
        pass
    elif select_college == 'Columbia':
        pass
    elif select_college == 'UPenn':
        pass
    elif select_college == 'Brown':
        College_text_data = open_data(select_college) # 선택한 대학의 정보가 담긴 txt 파일을 불러오고
        re_mjr = open_major_data(select_college, select_major) # 선택한 대학과 전공의 정보를 불러와서
        gen_keywd_college = general_keywords(College_text_data) # 키워드 추출하여 대학정보 WordCloud로 구현
        gen_keywd_college_major = general_keywords(re_mjr) # 키워드 추출하여 대학의 전공 WordCloud로 구현
    elif select_college == 'Cornell':
        pass
    elif select_college == 'Dartmouth':
        pass
    elif select_college == 'UChicago':
        pass
    elif select_college == 'Northwestern':
        pass
    elif select_college == 'Duke':
        pass
    elif select_college == 'Johns Hopkins':
        pass
    elif select_college == 'UCLA':
        pass
    elif select_college == 'UC Berkeley':
        pass
    elif select_college == 'Carnegie Mellon':
        pass
    elif select_college == 'Emory':
        pass
    elif select_college == 'Georgetown':
        pass
    elif select_college == 'UCLA':
        pass
    elif select_college == 'Emory':
        pass
    elif select_college == 'Caltech':
        pass
    elif select_college == 'USC':
        pass
    elif select_college == 'Georgetown':
        pass
    elif select_college == 'Willams':
        pass
    elif select_college == 'Swarthmore':
        pass
    elif select_college == 'Amherst':
        pass
    else:
        pass
    


    # prompt 에 해당하는 sentiments와 관련한 감정과, 입력한 에세이에서 추출한 감정들이 얼마나 일치 비율 계산하기
    # 에세이에서 추출 분석한 감정 리스트
    get_sents_from_essay = Prompt_Oriented_Sentiments_analysis(essay_input)
    # 선택한 해당 prompt의 감정 리스트 : pmt_sent_re
    pmt_snet_re_num = len(pmt_sent_re) # 선택한 prompt에서 추출한 감정 수
    cnt = 0
    for i in pmt_sent_re:
        if i in get_sents_from_essay[1]:
            cnt += 1
    
    cnt_re = cnt


    ############ - 에세이 입력 구간을 분리하여 감성분석 시작 - ##########

    def parted_Sentiments_analysis(essay_input):
        ########## 여기서는 최초 입력 에세이를 적용한다. input_text !!!!!!!!
        re_text = essay_input.split(".")

        #데이터 전처리 
        def cleaning(datas):
            fin_datas = []
            for data in datas:
                # 영문자 이외 문자는 공백으로 변환
                only_english = re.sub('[^a-zA-Z]', ' ', data)
                # 데이터를 리스트에 추가 
                fin_datas.append(only_english)
            return fin_datas

        texts = cleaning(re_text)
        texts_40 = texts[:int(round(len(texts)*0.4,0))] # 40% 앞부분 추출
        texts_60 = texts[int(round(len(texts)*0.4,0)):] # 60%은 뒷부분 추출

        def get_emo_text_ratio(text_input_list):
            #분석된 감정만 추출
            emo_re = goemotions(texts)

            emo_all = []
            for list_val in range(0, len(emo_re)):
                #print(emo_re[list_val]['labels'],emo_re[list_val]['scores'])
                #mo_all.append((emo_re[list_val]['labels'],emo_re[list_val]['scores'])) #KEY, VALUE만 추출하여 리스트로 저장
                #emo_all.append(emo_re[list_val]['scores'])
                emo_all.append((emo_re[list_val]['labels']))
                
            #추출결과 확인 
            # emo_all

            # ['sadness'],
            #  ['anger'],
            #  ['admiration', 'realization'],
            #  ['admiration', 'disappointment'],
            #  ['love'],
            #  ['sadness', 'neutral'],
            #  ['realization', 'neutral'],
            #  ['neutral'],
            #  ['optimism'],
            #  ['neutral'],
            #  ['excitement'],
            #  ['neutral'],
            #  ['neutral'],
            #  ['caring'],
            #  ['gratitude'],
            #  ['admiration', 'approval'], ...

            from pandas.core.common import flatten #이중리스틀 FLATTEN하게 변환
            flat_list = list(flatten(emo_all))

            # ['neutral',
            #  'neutral',
            #  'sadness',
            #  'anger',
            #  'admiration',
            #  'realization',
            #  'admiration',
            #  'disappointment',


            #중립적인 감정을 제외하고, 입력한 문장에서 다양한 감정을 모두 추출하고 어떤 감정이 있는지 계산해보자
            unique = []
            for r in flat_list:
                if r == 'neutral':
                    pass
                else:
                    unique.append(r)

            #중립감정 제거 및 유일한 감정값 확인
            #unique
            unique_re = set(unique) #중복제거

            ############################################################################
            # 글에 표현된 감정이 얼마나 다양한지 분석 결과!!!¶
            # print("====================================================================")
            # print("에세이에 표현된 다양한 감정 수:", len(unique_re))
            # print("====================================================================")

            #분석가능한 감정 총 감정 수 - Bert origin model 적용시 28개 감정 추출돰
            total_num_emotion_analyzed = 28

            # 감정기복 비율 계산 !!!
            result_emo_swings =round(len(unique_re)/total_num_emotion_analyzed *100,1) #소숫점 첫째자리만 표현
            # print("문장에 표현된 감정 비율 : ", result_emo_swings)
            # print("====================================================================")

            # unique_re : 에세이에서 분석 추출한 감정   ====> 이것이 중요한 값임
            return unique_re

        result_of_each_emotion_analysis_1 = get_emo_text_ratio(texts_40)
        result_of_each_emotion_analysis_2 = get_emo_text_ratio(texts_60)

        return result_of_each_emotion_analysis_1, result_of_each_emotion_analysis_2

        ############ - 구간분리 감성분석 결과 끝 - ##############


    # Social Awareness 부분에서 에세이 구간별 감성 분석 적용 부분 [:16], [17:]
    # pmt_sent_etc_re[:16] # 1차 감성부분 초반 40% (단어수따라 틀림), 총점의 40%: anger, annoyance, disapproval, disappointment, disgust, fear, grief, nervousness, sadness, surprise, remorse, curiosity, embarrassment
    # pmt_sent_etc_re[17:] # 2차 감성부분 후반 60% (단어수따라 틀림), 총점의 60%: admiration, approval, caring, joy, gratitude, optimism, realization, relief

    ### 문장 구간 분리하여 대표 감성 추출할 것 ###
    sent_parted_re = parted_Sentiments_analysis(essay_input)
    # 초반 40% 구간의 대표감성분석 결과
    sent_pre_40_re = sent_parted_re[0]
    # 일치비율 계산
    # 전반 40% 구간에 해당하는 감성정보값(리스트) : pmt_sent_etc_re[:16]
    # 비교
    s_40_cnt= 0
    for ittm in sent_pre_40_re:
        if ittm in pmt_sent_etc_re[:16]: # 전반 40% 구간에 일치하는 감성이 있다면,
            s_40_cnt += 1 # 카운트하고, 

    if s_40_cnt == 0: # 일치 하는 감성정보가 없다면,
        sent_comp_ratio_40 = 0
    else: # 있다면,
        sent_comp_ratio_40 = round(s_40_cnt / len(pmt_sent_etc_re[:16]) * 100, 2) * 0.4 # 전반 40% 구간에서 일치하는 감성의 선택한 프롬프트감성과의 비교결과 포함 비율을 계산하고, 가중치 적용(0.4)


    # 후반 60% 구간의 대표감성분석 결과
    sent_pre_60_re = sent_parted_re[1]
    # 일치비율 계산
    # 후반 60% 구간에 해당하는 감성정보값(리스트) : pmt_sent_etc_re[17:]
    s_60_cnt= 0
    for ittm_ in sent_pre_60_re:
        if ittm_ in pmt_sent_etc_re[17:]: # 후반 60% 구간 리스트에 일치하는 감성이 있다면,
            s_60_cnt += 1 # 카운트하고, 
    
    if s_60_cnt == 0: # 일치 하는 감성정보가 없다면,
        sent_comp_ratio_60 = 0
    else:
        sent_comp_ratio_60 = round(s_60_cnt / len(pmt_sent_etc_re[17:]) * 100, 2) * 0.6

    # 비율 적용한 최종 값
    fin_re_sentiments_analysis = sent_comp_ratio_40 + sent_comp_ratio_60
    print('fin_re_sentiments_analysis:', fin_re_sentiments_analysis)

    # 일치비율 계산
    sent_comp_ratio_origin = round(cnt_re / pmt_snet_re_num * 100, 2)


    def calculate_score(sent_comp_ratio):
        if sent_comp_ratio >= 80:
            result_pmt_ori_sentiments = 'Supurb'
        elif sent_comp_ratio >= 60 and sent_comp_ratio < 80:
            result_pmt_ori_sentiments = 'Strong'
        elif sent_comp_ratio >= 40 and sent_comp_ratio < 60:
            result_pmt_ori_sentiments = 'Good'
        elif sent_comp_ratio >= 20 and sent_comp_ratio < 40:
            result_pmt_ori_sentiments = 'Mediocre'
        else: #sent_comp_ratio < 20
            result_pmt_ori_sentiments = 'Lacking'
        return result_pmt_ori_sentiments


    result_pmt_ori_sentiments_of_social_issue = calculate_score(fin_re_sentiments_analysis) # Social issues: contribution & solution 부분의  Prompt Oriented Sentiments -- 분리적용한 부분
    result_pmt_ori_sentiments = calculate_score(sent_comp_ratio_origin)
    
    

    # 아래 코드는 사용하지 않음, 대신 sentence_similiarity.py 코드를 사용함
    # college_dept text data와 입력한 에세이의 데이터를 비교하여 유사도를 추출한다. ex) result ==> 're_coll_n_dept_fit': {0: 1.0}}
    # re_coll_n_dept_fit = college_n_dept_fit(College_text_data, coll_supp_essay_input_data)

    #######################################################################################
    # sentence_similiarity.py 코드
    ### 결과해석 ###
    # coll_dept_result : College & Department Fit ex)Weak, 생성한 문장
    # mjr_fit_result : Major Fit ex)Weak, 생성한 문장
    # TopComment : 첫번째 Selected Prompt 에 의한 고정 문장 생성

    # PmtOrientedSentments_result : 감성분석결과
        # counter : 선택한 prompt에 해당하는 coll supp essay의 대표적 감성 5개중 일치하는 상대적인 총 개수
        # matching_sentment : 매칭되는 감성 추출값
        # matching_ratio : 매칭 비율
        # match_result : 감성비교 최종 결과 산출

    # PmtOrientedSentments_result[3] : 최종 감성 상대적 비교 결과
    # overall_drft_sum : overall sum score(계산용 값)
    # overall_reault : Overall 최종 산출값
    #######################################################################################

    re_coll_n_dept_fit = sent_sim_analysis_with_bert_summarizer(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data)

    # major fit result  --> 이 값을 intellectualEngagement.py의 def intellectualEnguagement(essay_input, input_mjr_score)에 입력해서 결과를 다시 가져옴
    mjr_fit_result_final = re_coll_n_dept_fit[7]

    # Comment Generation
    # 입력 1번째 : score 는 각 입력부분에 해당하는 점수
    # 입력 2번째 : input_value는 4개중 하나로 major, general academic knowledge, pmt oriented sentments, intellectual enguagement 에서 1개 택 1
    # 입력 3번째 : input_mjr_score 는 sentence_similaty에서 에세이와 학교정보와의 전공 매칭 결과를 가져와서 입력
    
    def commentsGen(score, input_value):
        if score >= 65: #Superb & Strong 
            if input_value == 'majorFitCmt':
                majorFitCmt = """Regarding your fit with the intended major, your knowledge of the discipline's intellectual concepts seems quite extensive."""
                gen_comment = majorFitCmt
            elif input_value == 'General_Academic_Knowledge_cmt':
                General_Academic_Knowledge_cmt = """Beyond your academic major, you seem to possess a versatile range of knowledge in various intellectual topics that would impress the reader."""
                gen_comment = General_Academic_Knowledge_cmt
            elif input_value == 'Prompt_Oriented_Sentiments':
                Prompt_Oriented_Sentiments = """Sentiment analysis shows that you demonstrate a healthy level of curiosity and grasp of the concepts."""
                gen_comment = Prompt_Oriented_Sentiments
            elif input_value == 'Intellectual_Engagement':
                Intellectual_Engagement = """Lastly, you seem to weave together seemingly distant topics and ideas successfully. Hence, your story looks quite original and versatile."""
                gen_comment = Intellectual_Engagement
            else:
                pass

        elif score >= 35 and score < 65:# Good
            if input_value == 'majorFitCmt':
                majorFitCmt = """Regarding your fit with the intended major, your knowledge of the discipline's intellectual concepts seems good."""
                gen_comment = majorFitCmt
            elif input_value == 'General_Academic_Knowledge_cmt':
                General_Academic_Knowledge_cmt = """Beyond your academic major, you seem to possess a fair amount of knowledge in various intellectual topics."""
                gen_comment = General_Academic_Knowledge_cmt
            elif input_value == 'Prompt_Oriented_Sentiments':
                Prompt_Oriented_Sentiments = """Sentiment analysis shows that you demonstrate a satisfactory level of curiosity and grasp of the concepts."""
                gen_comment = Prompt_Oriented_Sentiments
            elif input_value == 'Intellectual_Engagement':
                Intellectual_Engagement = """Lastly, you seem to demonstrate your thought process in some detail. Consider adding more of your own analysis and insights to make your ideas sound more versatile."""
                gen_comment = Intellectual_Engagement
            else:
                pass

        else: # score < 35  : Mediocre & Weak
            if input_value == 'majorFitCmt':
                majorFitCmt = """Regarding your fit with the intended major, your knowledge of the discipline's intellectual concepts seems lacking."""
                gen_comment = majorFitCmt
            elif input_value == 'General_Academic_Knowledge_cmt':
                General_Academic_Knowledge_cmt = """Also, it seems that your knowledge is more focused on the area of your academic major, possibly lacking some diversity."""
                gen_comment = General_Academic_Knowledge_cmt
            elif input_value == 'Prompt_Oriented_Sentiments':
                Prompt_Oriented_Sentiments = """Sentiment analysis shows you may consider adding in more phrases that highlight your curiosity and the lessons you draw from the topic."""
                gen_comment = Prompt_Oriented_Sentiments
            elif input_value == 'Intellectual_Engagement':
                Intellectual_Engagement = """Lastly, please consider elaborating further on the thought process by adding your own analysis and insights to emphasize the level of intellectual engagement."""
                gen_comment = Intellectual_Engagement
            else:
                pass

        # 조건에 의한 문장 추출
        return gen_comment


    ###############  General Academic Knowledge ############# 
    # 첫번째 고정문장 
    fixedTopCmt = """An intellectual interest essay may deal with any topic as long as it demonstrates the writer’s knowledge, analytical thinking, and creativity. Nonetheless, experts advise that displaying the depth of knowledge in your intended major area in a curious and insightful manner could provide a more precise focal point for the reviewer. Engaging ideas can be demonstrated through a healthy level of cohesion and academically-oriented verbs, while how you connect the dots between seemingly distant ideas can show how original your thoughts are."""

    GAC_re = GeneralAcademicKnowledge(essay_input) # 이 함수 값에서
    GAC_Sentences = GAC_re[6] # 6. totalSettingSentences : academic 단어가 포함된 모든 문장을 추출 -------> 웹에 표시할 문장(아카데믹 단어가 포함된 문장)
    GAC_Words = GAC_re[11] # 11. topic_academic_word --------> 이 값을 가지고 비교할 것 - 웹에 표시할 단어들(아카데믹 단어)
    GAC_words_usage_rate = GAC_re[12]   # 12. topic_academic_word_counter  ---------> 이 값을 가지고 비교할 것 - 아카데믹 단서 사용 비율
    GAK_rate = GAC_re[13] # 14. GAK_rate : General Academic Knowledge ----> 웹에 적용할 부분 "Supurb ~ Weak " 중에서 하나가 나옴
    GAC_Topics_score = GAC_re[15] #General Academic Topics Score --> intellectual interest 의 overall score 계산을 위해서 값 추출
    extracted_topics_of_essay = GAC_re[16] # 에세이에서 추출한 토픽 -> 이것을 topic uniquness 분석에 활용할거임
    print('GAC_Topics_score ==========================> ', GAC_Topics_score )

    # Topic uniquenss 분석
    result_topic_uniqueness = []
    result_topic_unique_score = []
    for tp_itm in extracted_topics_of_essay[:3]:# 추출한 토픽중 3개만 분석하기
        get_topic_uniqueness_re = google_search_result(tp_itm) # 구글 검색하여 결과 추출
        get_topic_uniqueness_re[0] # 추출한 토픽의 uniqueness -- 'very unique'
        get_topic_uniqueness_re[1] # 추출한 토픽의 스코어 - 90점
        result_topic_uniqueness.append(get_topic_uniqueness_re[0])
        result_topic_unique_score.append(get_topic_uniqueness_re[1])

    # Topic uniqueness 토픽 3개를 반영한 최종 점수 계산
    topic_uniquness_fin_score = round(sum(result_topic_unique_score) / len(result_topic_unique_score), 2)

    # Topic uniqueness의 평균 값을 산출하기 위해서 topic_uniquness_fin_score을 가지고 common, unique, very unique를 계산
    if topic_uniquness_fin_score > 90:
        tp_uniqueness = 'Very Unique'
        tp_uniqueness_5dv = 'Supurb'
    elif topic_uniquness_fin_score >= 60 and topic_uniquness_fin_score > 30:
        tp_uniqueness = 'Unique'
        tp_uniqueness_5dv = 'Good'
    else: # topic_uniquness_fin_score <= 30
        tp_uniqueness = 'Common'
        tp_uniqueness_5dv = 'Mediocre'


    # Topic knowledge 점수 계산
    result_topic_knowledge =[]
    for ets_itm in extracted_topics_of_essay[:3]: # 추출한 토픽 3개만 분석하기(시가간이 많이 걸림- 3개는 평균 39개의 웹페이지 분석해야 함)
        result_of_srch = google_search_result_tp_knowledge(ets_itm) # 각 토픽별로 관련 웹검색하여 단어 추출
        result_topic_knowledge.append(result_of_srch) # 추출 리스트 저장
    print('result_topic_knowledge:', result_topic_knowledge)

    # Topic knowledge결과 비교하기 : 전체 추출 리스트와 추출한 토픽들의 포함 비율 계산하기
    match_topic_words = 0
    for ext_itttm in extracted_topics_of_essay:
        if ext_itttm in result_topic_knowledge: # 토픽이 리스트안에 있다면! 카운트한다.
            match_topic_words += 1
    print('match_topic_words:', match_topic_words)

    if match_topic_words != 0: # 매칭되는 토픽이 있다면, 검색을 통해 수집된 정보에서 매칭 토픽의 포함 비율을 계산해본다. 예를 들어 일정 기준 이상이면 strong.. 등으로 표현하면 된다.
        get_topic_knowledge_ratio = round(match_topic_words / len(result_topic_knowledge) * 100, 2)
        print('get_topic_knowledge_ratio:', get_topic_knowledge_ratio)
        if get_topic_knowledge_ratio >= 10: #10% 이상이면 ================> 중요! 이 값은 결과값을 보면서 보정해야 함(현재는 임의값 적용)
            fin_topic_knowledge_score = 'Supurb'
        elif get_topic_knowledge_ratio >= 5 and get_topic_knowledge_ratio < 10: #================> 중요! 이 값은 결과값을 보면서 보정해야 함(현재는 임의값 적용)
            fin_topic_knowledge_score = 'Strong'
        elif get_topic_knowledge_ratio >= 3 and get_topic_knowledge_ratio < 5: #================> 중요! 이 값은 결과값을 보면서 보정해야 함(현재는 임의값 적용)
            fin_topic_knowledge_score = 'Good' 
        else:
            fin_topic_knowledge_score = 'Mediocre'
    else: # match_topic_words = 0 매칭하는 값이 0이면=================>>>>>>>>>>>> !!! 결과값 재획인 해야 함!!!
        fin_topic_knowledge_score = 'Lacking'
        get_topic_knowledge_ratio = 0

    #supurb ~ lacking 을 숫자로 된 점수로 변환
    def text_re_to_score(input):
        if input == 'Supurb':
            tp_knowledge_re = 90
        elif input == 'Strong':
            tp_knowledge_re = 75
        elif input == 'Good':
            tp_knowledge_re = 65
        elif input == 'Mediocre':
            tp_knowledge_re = 40
        else: #input == 'Lacking'
            tp_knowledge_re = 10
        return tp_knowledge_re
    # supurb ~ lacking 을 숫자로 된 점수로 변환
    tp_kwlg_result = text_re_to_score(fin_topic_knowledge_score)
    # print('tp_kwlg_result:', tp_kwlg_result)




    # initiative_engagement_contribution
    ini_engage_re = initiative_engagement_contribution(essay_input)
    ini_engage_5div_re = ini_engage_re[2][0] # supurb ~ lacking 로 결과나옴
    ini_engage_words = ini_engage_re[1] # initiative_engagement_contribution 관련 단어들로 웹에 표시
    ini_engage_fin_score_re = ini_engage_re[2][1] # 점수로 계산됨 ---> for overall score 

    social_aware_re = social_awareness_analysis(essay_input)
    social_aware_5div_re = social_aware_re[2][0] # supurb ~ lacking 로 결과나옴
    social_aware_words = social_aware_re[1] # Social Awareness 관련 단어들로 웹에 표시
    social_aware_fin_score_re = social_aware_re[2][1]

    # Social issues: Contribution & soluotion ==> overall score
    social_iss_cont_sln_overall_score = float(social_aware_fin_score_re) * 0.3 + float(ini_engage_fin_score_re) * 0.3 + float(topic_uniquness_fin_score) * 0.1 + tp_kwlg_result * 0.1
    
    # Social issues: Contribution & soluotion ==> 문장생성
    social_comment_fixed_achieve = """Powerful essays about social issues involve multiple elements. Your knowledge of the given issue and activism will demonstrate social awareness. Meanwhile, you should be emotionally engaged, especially with the social problems that made you angry or disappointed. Then, your realization of the issues should be backed up by your action – to bring about changes."""
    def social_gen_comment_achievement(input_score, type):
        if input_score == 'Supurb' or input_score == 'Strong':
            if type == 'social_awareness':
                comment_achieve= """Your essay seems to demonstrate your wealth of knowledge in social issues and activism."""
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Also, it is clear that you engage in this particular issue emotionally, and"""
            elif type == 'initiative_eng':
                comment_achieve = """your story seems to demonstrate a high level of effort of rmaking improvements.""" 
            elif type == 'topic_uniqueness':
                comment_achieve = """The topics in your essay seem unique, and readers may find them intriguing.""" 
            elif type == 'topic_knowledge':
                comment_achieve = """In terms of your knowledge of the topic, you seem to be very knowledgeable.""" 
            else:
                pass
        elif input_score == 'Good':
            if type == 'social_awareness':
                comment_achieve= '''Your essay seems to demonstrate your knowledge of social issues and activism.'''
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Also, you seem to engage in this particular issue emotionally, and"""
            elif type == 'initiative_eng':
                comment_achieve = """your story seems to demonstrate a satisfactory level of effort for making improvements.""" 
            elif type == 'topic_uniqueness':
                comment_achieve = """The topics in your essay seem somewhat unique, and you may consider finding more exciting topics.""" 
            elif type == 'topic_knowledge':
                comment_achieve = """In terms of your knowledge of the topic, you seem to be somewhat knowledgeable.""" 
            else:
                pass
        else: #input score == 'Mediocre' or input_score == 'Weak'
            if type == 'social_awareness':
                comment_achieve= '''Your essay may need some improvements in displaying your knowledge of social issues and activism.'''
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Also, your emotional engagement with this particular issue is somewhat weak, while"""
            elif type == 'initiative_eng':
                comment_achieve = """your story may need to demonstrate a higher amount of effort for making improvements.""" 
            elif type == 'topic_uniqueness':
                comment_achieve = """The topics in your essay seem somewhat familiar. Hence, you may consider finding less generic issues while adding more detail.""" 
            elif type == 'topic_knowledge':
                comment_achieve = """In terms of your knowledge of the topic, you may need to include more details.""" 
            else:
                pass
        return comment_achieve


    #문장생성
    gen_sent_social_awareness = social_gen_comment_achievement(social_aware_5div_re, 'social_awareness')
    gen_pmt_ori_sent_social = social_gen_comment_achievement(result_pmt_ori_sentiments, 'pmt_ori_sentiment')
    gen_initiative_engs = social_gen_comment_achievement(ini_engage_5div_re, 'initiative_eng')
    gen_topic_uniqueness = social_gen_comment_achievement(tp_uniqueness_5dv, 'topic_uniqueness')
    gen_topic_knowledge = social_gen_comment_achievement(fin_topic_knowledge_score, 'topic_knowledge')




    #문장생성 !!!!! mjr_score 값을 계산해야 함
    mjr_score = mjr_fit_result_final
    general_aca_knowledge_score = GAC_Topics_score
    # 감성 정보 계산 값
    sentiments_score = re_coll_n_dept_fit[3][2]
    # intellectualenguagement  값 계산
    from intellectualEngagement import intellectualEnguagement
    intel_eng_score_all = intellectualEnguagement(essay_input)
    intel_eng_score = intel_eng_score_all[0] # 위 함수의 첫번재 값이 intell Eng Score 임
    
    # 이하 값은 return으로 출력해서 문장을 생성한다.
    mjr_comment_re = commentsGen(mjr_score, 'majorFitCmt')
    general_aca_comment_re = commentsGen(general_aca_knowledge_score, 'General_Academic_Knowledge_cmt')
    pmt_ori_sentiments_re = commentsGen(sentiments_score, 'Prompt_Oriented_Sentiments')
    intellectual_eng_re = commentsGen(intel_eng_score, 'Intellectual_Engagement')


    ## meaningfulExperienceLessonLearned.py 의 코멘트 생성 부분
    meanful_result = MeaningFullExpreenceLessonLearened(essay_input)
    ### MeaningFullExpreenceLessonLearened(essay_input)의 return 값 설명 ###
        # 0. overall_result_fin_re : overall score
        # 1. KeyKiterElement_score
        # 2. PromptOriented_score
        # 3. Originality_score
        # 4. Perspective_score

        # 5. plot_n_conflict_word_for_web : 웹에 표시되는 단어 리스트
        # 6. characgter_words_for_web : 웹에 표시되는 단어 리스트
        # 7. setting_words_list : 웹에 표시되는 단어 리스트
        # 8. perspective_analysis_result_for_web :  웹에 표시되는 단어+문장 리스트

        # 9. fixed_top_comment :  코멘트 생성
        # 10. KLE_comment : 코멘트 생성
        # 11. keylitElemt_comment : 코멘트 생성
        # 12. Originality_comment : 코멘트 생성
        # 13. perspective_comment : 코멘트 생성


    achievement_result = get_achievement_you_are_proud_of(essay_input)
    ### achievement_result 결과값 해석 ###   --- achievementYouAreProud.py
        # 0. achievement_result[0]   : in_result : 최종 결과로 5가지 척도로 계산됨
        # 1. achievement_result[1]   :get_words_ratio : 입력에세이의 토픽과 비교할 단어가 얼마나 일치하는지에 대한 비율 계산 결과
        # 2. achievement_result[2]   :pmt_ori_keyword : Prompt Oriented Keywords 추출
        # 3. achievement_result[3]   :fin_initiative_enguagement_ratio : initiative_enguagement 가 에세이이 포함된 비율
        # 4. iachievement_result[4]   :nitiative_enguagement_result : initiative_enguagement가 합격생 평균에 비교하여 얻은 최종 값

    #achievement you are proud of  - overall 결과 계산하기
    def cal_sore(input_5d_value):
        if input_5d_value == 'Supurb':
            get_score = 100
        elif input_5d_value == 'Strong':
            get_score = 80
        elif input_5d_value == 'Good':
            get_score = 60
        elif input_5d_value == 'Mediocre':
            get_score = 40
        elif input_5d_value == 'Lacking':
            get_score = 20
        else:
            pass
        return get_score

    prompt_ori_keywds = cal_sore(achievement_result[0]) #20%
    prompt_ori_sentiments = cal_sore(result_pmt_ori_sentiments)# 40%
    initiative_eng = cal_sore(achievement_result[4]) # 30%
    
    # overall achievement 최종값 계산!!!!
    overall_of_achievement_you_are_prooud_of = prompt_ori_keywds * 0.2 + prompt_ori_sentiments * 0.4 + initiative_eng * 0.3

    comment_fixed_achieve = """Writing about an achievement you are proud of entails multiple elements. You may consider including words that are closely related to a noteworthy achievement. Usually, concepts like leadership, cooperation, overcoming a hardship, triumph, and more would suit such a topic. Also, there should be sentiments that convey a sense of pride, realization, appreciation, and determination while highlighting the course of your action to reach the end result."""
    def gen_comment_achievement(input_score, type):
        if input_score == 'Supurb' or input_score == 'Strong':
            if type == 'pmt_ori_keywd':
                comment_achieve= '''Your essay seems to contain a robust set of words associate with achievement.'''
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """In addition, you seem to display strong sentiments that represent the outcome you are proud of."""
            elif type == 'initiative_eng':
                comment_achieve = """Your story seems to demonstrate a high level of effort and leadership, which fits the prompt's qualities very well.""" 
            else:
                pass
        elif input_score == 'Good':
            if type == 'pmt_ori_keywd':
                comment_achieve= 'Your essay seems to contain a sufficient amount of words associate with achievement.'
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """In addition, you seem to display adequate sentiments that constitute the outcome you are proud of."""
            elif type == 'initiative_eng':
                comment_achieve = """Your story seems to demonstrate a satisfactory level of effort and leadership, which fits the prompt's qualities."""
            else:
                pass
        else: #input score == 'Mediocre' or input_score == 'Weak'
            if type == 'pmt_ori_keywd':
                comment_achieve = """Your essay seems to contain an insufficient amount of words associate with achievement."""
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """In addition, you seem to be lacking the sentiments that constitute the outcome you are proud of."""
            elif type == 'initiative_eng':
                comment_achieve = """Your story seems to demonstrate an insufficient level of effort and leadership, which fits the prompt's qualities."""
            else:
                pass

        return comment_achieve

    # achievement 문장 생성
    gen_achi_pmt_ori_keywd =  gen_comment_achievement(achievement_result[0], 'pmt_ori_keywd')
    gen_arch_pmt_ori_sentiment =  gen_comment_achievement(result_pmt_ori_sentiments, 'pmt_ori_sentiment')
    gen_arch_initiative_eng = gen_comment_achievement(achievement_result[4], 'initiative_eng')

    comments_achievement = [comment_fixed_achieve, gen_achi_pmt_ori_keywd, gen_arch_pmt_ori_sentiment, gen_arch_initiative_eng]


    summerActivity_re = SummerActivity(essay_input)
    popular_summer_program = summerActivity_re[0]
    name_pop_summ_prmg = summerActivity_re[1]
    pop_sum_prmg_score = summerActivity_re[2]

    sum_act_ini_eng = summer_activity_initiative_engagement(essay_input)
    sum_act_ini_eng_score = sum_act_ini_eng[0]
    sum_act_ini_eng_words = sum_act_ini_eng[1]



    ### +++ 실행결과 설명 +++ ###
    # 0. gen_keywd_college : 선택한 대학의 General Keywords on college로 wordcloud로 출력됨
    # 1. gen_keywd_college_major : 선택 대학의 전공에 대한 keywords 를 WrodCloud 로 출력
    # 2. intended_mjr : intended major
    # 3. pmt_sent_etc_re : 선택한 prompt 질문
    # 4. prompt_type_sentence : 선택한 prompt에 해당하는 질문 문장 전체
    # 5. pmt_sent_re : 선택한 prompt에 해당하는 sentiment 리스트
    # 6 'result_pmt_ori_sentiments' : result_pmt_ori_sentiments, # prompt oriented keywords 값으로 5점척도임(Supurb~lacking) --> 웹이 표시함
    # 7. re_coll_n_dept_fit : sentence_similiarity.py 코드의 결과값임
    # 8. GAC_Sentences = GAC_re[6] totalSettingSentences : academic 단어가 포함된 모든 문장을 추출 -------> 웹에 표시할 문장(아카데믹 단어가 포함된 문장)
    # 9. GAC_Words = GAC_re[11] topic_academic_word --------> 이 값을 가지고 비교할 것 - 웹에 표시할 단어들(아카데믹 단어)
    # 10. GAC_words_usage_rate = GAC_re[12]   topic_academic_word_counter  ---------> 이 값을 가지고 비교할 것 - 아카데믹 단서 사용 비율
    # 11. GAK_rate = GAC_re[13] # GAK_rate : General Academic Knowledge ----> 웹에 적용할 부분 "Supurb ~ Weak " 중에서 하나가 나옴
    # 12. fixedTopCmt : 첫번째 고정문장 생성
    # 13. mjr_comment_re : 전공적합성 문장생성 부분
    # 14. general_aca_comment_re : general academic knowledge 문장생성 부분
    # 15. pmt_ori_sentiments_re : sentiments 문장생성 부분
    # 16. intellectual_eng_re : intellectual enguagement 문장생성 부분
    # 17. meanful_result : MeaningFullExpreenceLessonLearened(essay_input)의 return 값
            #- MeaningFullExpreenceLessonLearened(essay_input)의 return 값 설명 ###
            # 0. overall_result_fin_re : overall score
            # 1. KeyKiterElement_score
            # 2. PromptOriented_score
            # 3. Originality_score
            # 4. Perspective_score

            # 5. plot_n_conflict_word_for_web : 웹에 표시되는 단어 리스트
            # 6. characgter_words_for_web : 웹에 표시되는 단어 리스트
            # 7. setting_words_list : 웹에 표시되는 단어 리스트
            # 8. perspective_analysis_result_for_web :  웹에 표시되는 단어+문장 리스트

            # 9. fixed_top_comment :  코멘트 생성
            # 10. KLE_comment : 코멘트 생성
            # 11. keylitElemt_comment : 코멘트 생성
            # 12. Originality_comment : 코멘트 생성
            # 13. perspective_comment : 코멘트 생성

            #### 이하 내용은 MeaningFullExpreenceLessonLearened(essay_input)의 return 값을 딕서녀러의 key : value 로 정리한 것

            # 'overall_score' : overall_score, 
            # 'KeyKiterElement_score' : KeyKiterElement_score, 
            # 'PromptOriented_score' : PromptOriented_score, 
            # 'Originality_score': Originality_score, 
            # 'Perspective_score': Perspective_score, 
            # 'plot_n_conflict_word_for_web' : plot_n_conflict_word_for_web, 
            # 'characgter_words_for_web' : characgter_words_for_web, 
            # 'setting_words_list' : setting_words_list, 
            # 'perspective_analysis_result_for_web' : perspective_analysis_result_for_web, 
            # 'fixed_top_comment' : fixed_top_comment, 
            # 'KLE_comment' : KLE_comment, 
            # 'keylitElemt_comment' : keylitElemt_comment, 
            # 'Originality_comment' : Originality_comment, 
            # 'perspective_comment' : perspective_comment

        # 18. achievement_result
            # 0. achievement_result[0]  : in_result : 최종 결과로 5가지 척도로 계산됨
            # 1. achievement_result[1]  : get_words_ratio : 입력에세이의 토픽과 비교할 단어가 얼마나 일치하는지에 대한 비율 계산 결과
            # 2. achievement_result[2]  : pmt_ori_keyword : Prompt Oriented Keywords 추출
            # 3. achievement_result[3]  : fin_initiative_enguagement_ratio : initiative_enguagement 가 에세이이 포함된 비율
            # 4. iachievement_result[4] : nitiative_enguagement_result : initiative_enguagement가 합격생 평균에 비교하여 얻은 최종 값
        # 19. overall_of_achievement_you_are_prooud_of # overall achievement 최종값  -- 웹에 표시
        # 20. result_topic_uniqueness # Topic uniqueness 추출결과 - 3개만 분석하고, 분석결과는 data/topic_search_result.xlsx 에 저장됨
        # 21. fin_topic_knowledge_score # Topic Knowledge Scofre


    data_result = {
        'gen_keywd_college' : gen_keywd_college, # 선택한 대학의 General Keywords on college로 wordcloud로 출력됨
        'gen_keywd_college_major' : gen_keywd_college_major, # 선택 대학의 전공에 대한 keywords 를 WrodCloud 로 출력
        'intended_mjr' : intended_mjr, #intended major
        'pmt_sent_etc_re' : pmt_sent_etc_re, #선택한 prompt 질문
        'prompt_type_sentence' : prompt_type_sentence, #선택한 prompt에 해당하는 질문 문장 전체
        'pmt_sent_re' : pmt_sent_re, # 선택한 prompt에 해당하는 sentiment 리스트

        #result_pmt_ori_sentiments는 다양한 prompt에서 모두 적용됨(공통적용)
        'result_pmt_ori_sentiments' : result_pmt_ori_sentiments, # prompt oriented keywords 값으로 5점척도임(Supurb~lacking) --> 웹이 표시함

        're_coll_n_dept_fit' : re_coll_n_dept_fit, # College & Dept.Fit으로 입력한 Supplyment Essay와 비교하여 적합성  TFD-IDF로 계산해볼 것(lexicon 사용하지 않고 계산하였음. 성능이 낮으면 lexicon 추가하여 계사할 거임)
        'GAC_Sentences' : GAC_re[6], # 6. totalSettingSentences : academic 단어가 포함된 모든 문장을 추출 -------> 웹에 표시할 문장(아카데믹 단어가 포함된 문장)
        'GAC_Words' : GAC_re[11], # 11. topic_academic_word --------> 이 값을 가지고 비교할 것 - 웹에 표시할 단어들(아카데믹 단어)
        'GAC_words_usage_rate' : GAC_re[12],   # 12. topic_academic_word_counter  ---------> 이 값을 가지고 비교할 것 - 아카데믹 단서 사용 비율
        'GAK_rate' : GAC_re[14], # 14. GAK_rate : General Academic Knowledge ----> 웹에 적용할 부분 "Supurb ~ Weak " 중에서 하나가 나옴
        'fixedTopCmt' : fixedTopCmt, # 첫번째 위치하는 고정문장 생성
        'mjr_comment_re' : mjr_comment_re, # intellectualEngagemnet 부분의 major fit comment
        'general_aca_comment_re' : general_aca_comment_re, # general academic knowledge 문장생성 부분
        'pmt_ori_sentiments_re' : pmt_ori_sentiments_re, # sentiments 문장생성 부분
        'intellectual_eng_re' : intellectual_eng_re, # intellectual enguagement 문장생성 부분
        'meanful_result' : meanful_result, # 이 부분은 위에 # 16. meanful_result : MeaningFullExpreenceLessonLearened(essay_input)의 return 값에 해당하는 부부으로 다수의 값이 계산된다. 코멘트까지 계산된다규~!
        'achievement_result' : achievement_result,
        'overall_of_achievement_you_are_prooud_of' : overall_of_achievement_you_are_prooud_of, # overall achievement 최종값  -- 웹에 표시
        'comments_achievement': comments_achievement, # Achievement 문장 생성 부분 총 4개
        'result_pmt_ori_sentiments_of_social_issue': result_pmt_ori_sentiments_of_social_issue, # Social issues: contribution & solution - Prompt Oriented Sentiments 계산 결과임 ---> 웹에 표시할 것
        'result_topic_uniqueness' : result_topic_uniqueness, # Topic uniqueness 추출결과 - 3개만 분석하고, 분석결과는 data/topic_search_result.xlsx 에 저장됨
        'extracted_topics_of_essay[:2]' : extracted_topics_of_essay[:2], # 에세에서 추출한 주요 토픽 3개, 그 이상을 보여주려면 extracted_topics_of_essay[:2~ 이상의 값을 넣으면 됨, 많이 넣으면 our of value 로 에러남옴] ---> 웹에 표시할 것
        
        'social_iss_cont_sln_overall_score': social_iss_cont_sln_overall_score, # Social issues: contribution & solution의 overall score
        'social_aware_5div_re' : social_aware_5div_re, # social awareness 의 결과 supurb ~ lacking 으로 나옴
        'social_aware_words' : social_aware_words,   # Social Awareness 관련 단어들로 웹에 표시
        # Prompt Oriented Sentiments -- esult_pmt_ori_sentiments_of_social_issue <- 이것이 계산 결과임
        'ini_engage_5div_re' : ini_engage_5div_re, #  Initiative, Engagement, & Contribution 의 결과로 supurb ~ lacking 으로 출력됨
        'ini_engage_words' : ini_engage_words, # Initiative, Engagement, & Contribution의 결과 관련 단어 ------> 웹에 표시함

        'tp_uniqueness': tp_uniqueness, # 추출한 토핌의 최종 평균 uniqueness 값  common, unique, very unique 중 1개가 추출됨
        'topic_uniquness_fin_score' : topic_uniquness_fin_score, # 추출한 토픽의 스코어 topic uniqueness final score

        'fin_topic_knowledge_score' : fin_topic_knowledge_score, # Topic Knowledge Score
        'get_topic_knowledge_ratio' : get_topic_knowledge_ratio, # 에세이주요토픽추출단어/구글 검색 추출단어리스트 * 100 을 계산하여 Topic knowledge의 비율 계산
        'social_comment_fixed_achieve' : social_comment_fixed_achieve, # 코멘트생성
        # 문장생성
        'gen_sent_social_awareness' : gen_sent_social_awareness,
        'gen_pmt_ori_sent_social' : gen_pmt_ori_sent_social,
        'gen_initiative_engs' : gen_initiative_engs,
        'gen_topic_uniqueness' : gen_topic_uniqueness,
        'gen_topic_knowledge' : gen_topic_knowledge,
        # summer activity
        'popular_summer_program' : popular_summer_program, # 5개의 값으로 추출됨
        'name_pop_summ_prmg' : name_pop_summ_prmg, # 추출된 summer activitie 로 웹에 출력됨. 없으면 출력안됨(당연히)
        'pop_sum_prmg_score' : pop_sum_prmg_score, # summer activity overall 값 게산하기 위한 결과임

        'sum_act_ini_eng_score' : sum_act_ini_eng_score,
        'sum_act_ini_eng_words' : sum_act_ini_eng_words,
    }

    return data_result



### 실행 ###
# input College Supp Essay 
essay_input = """I inhale deeply and blow harder than I thought possible, pushing the tiny ember from its resting place on the candle out into the air. The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

# 입력값은 대학, 전공 ex) 'why_us', 'Browon', 'African Studies', text_input
# 입력값은 대학, 전공 ex) 'Intellectual interest', 'Browon', 'African Studies', text_input
#sc_re = selected_college('Why us', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
#sc_re = selected_college('Intellectual interest', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
#sc_re = selected_college("Meaningful experience & lesson learned", 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
#sc_re = selected_college("Achievement you are proud of", 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
#sc_re = selected_college("Social issues: contribution & solution", 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
sc_re = selected_college("Summer activity", 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
print('최종결과:', sc_re)

