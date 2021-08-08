# upgrade... 2021_07_28

#conflict
import pickle
import nltk


##########  key_value_print
##########  key: value 형식으로 나뉨 
def key_value_print (dictonrytemp) : 
    print("#"*100)
    print()
    for key in dictonrytemp.keys() : 
        print(key,": ",dictonrytemp[key])
    print()
    print("#"*100)


# 다운로드 이미 완료, 실행시 사용하지 않음
# punkt = nltk.download('punkt')
# with open('nltk_punkt.pickle', 'wb') as f:
#     pickle.dump(punkt, f, pickle.HIGHEST_PROTOCOL)

with open('./accu_ps/plot_conflict/nltk_punkt.pickle', 'rb') as f:
    punkt = pickle.load(f)


# 다운로드 이미 완료, 실행시 사용하지 않음
# vader_lexicon = nltk.download('vader_lexicon')
# with open('vader_lexicon.pickle', 'wb') as f:
#     pickle.dump(punkt, f, pickle.HIGHEST_PROTOCOL)

with open('./accu_ps/plot_conflict/vader_lexicon.pickle', 'rb') as f:
    vader_lexicon = pickle.load(f)


# 다운로드 이미 완료, 실행시 사용하지 않음
# averaged_perceptron_tagger  = nltk.download('averaged_perceptron_tagger')
# with open('averaged_perceptron_tagger.pickle', 'wb') as f:
#     pickle.dump(punkt, f, pickle.HIGHEST_PROTOCOL)
    
with open('./accu_ps/plot_conflict/averaged_perceptron_tagger.pickle', 'rb') as f:
    averaged_perceptron_tagger = pickle.load(f)

from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import pandas as pd
from pandas import DataFrame as df
from mpld3 import plugins, fig_to_html, save_html, fig_to_dict
from tqdm import tqdm
import numpy as np
import json
from tensorflow.keras.preprocessing.text import text_to_word_sequence

#character, setting
import numpy as np
import gensim
import nltk

#다운로드 이미 완료, 실행시 사용하지 않음
# stopwords = nltk.download('stopwords')
# with open('stopwords.pickle', 'wb') as f:
#     pickle.dump(stopwords, f, pickle.HIGHEST_PROTOCOL)
    
with open('./accu_ps/plot_conflict/stopwords.pickle', 'rb') as f:
    stopwords = pickle.load(f)

from nltk.corpus import stopwords
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


from transformers import BertTokenizer
from accu_ps.data.model import BertForMultiLabelClassification
from accu_ps.data.multilabel_pipeline import MultiLabelPipeline

from collections import Counter


# from model import BertForMultiLabelClassification
# from multilabel_pipeline import MultiLabelPipeline





# # save tokenizer ---> 계산속도 줄이기위해서 미리 저장, 저장했기 때문에 실행시는 사용하지 않음
# with open('data_tokenizer.pickle', 'wb') as f:
#     pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)

# # save model ---> 계산속도 줄이기위해서 미리 저장, 저장했기 때문에 실행시는 사용하지 않음

# with open('/var/www/html/essayfit/accu_ps/plot_conflict/data_model.pickle', 'wb') as g:
#     pickle.dump(model, g, pickle.HIGHEST_PROTOCOL)


# open tokenizer
with open('/var/www/html/essayfit/accu_ps/plot_conflict/data_tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)


with open('/var/www/html/essayfit/accu_ps/plot_conflict/data_model.pickle', 'rb') as g:
    model = pickle.load(g)


goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
)


#데이터 전처리 
def cleaning(datas):

    fin_datas = []

    for data in datas:
        # 영문자 이외 문자는 공백으로 변환
        only_english = re.sub('[^a-zA-Z]', ' ', data)
    
        # 데이터를 리스트에 추가 
        fin_datas.append(only_english)

    return fin_datas


def ai_plot_conf(essay_input_):
    #1.input essay
    input_text = essay_input_

    #########################################################################

    #2.유사단어를 추출하여 리스트로 반환
    def conflict_sim_words(text):

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

        #표현하는 단어들을 리스트에 넣어서 필터로 만들고
        confict_words_list = ['clash', 'incompatible', 'inconsistent', 'incongruous', 'opposition', 'variance','vary', 'odds', 
                                'differ', 'diverge', 'disagree', 'contrast', 'collide', 'contradictory', 'incompatible', 'conflict',
                                'inconsistent','irreconcilable','incongruous','contrary','opposite','opposing','opposed',
                                'antithetical','clashing','discordant','differing','different','divergent','discrepant',
                                'varying','disagreeing','contrasting','at odds','in opposition','at variance' ]
        
        ####문장에 list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
        
        #우선 토큰화한다.
        retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
        token_input_text = retokenize.tokenize(essay_input_corpus)
        
        #print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
        #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
        conflict_text_org = []
        for k in token_input_text:
            for j in confict_words_list:
                if k in j:
                    conflict_text_org.append(j)
        
        #print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
        
        filtered_chr_text_ = set(conflict_text_org) #중복제거
        filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
        # print("#"*100)
        # print("token_input_text:",token_input_text)
        # print()
        # print("filtered_chr_text__:",filtered_chr_text__)
        # print("#"*100)
        #print (filtered_chr_text__) # 중복값 제거 확인
        

        ext_sim_words_key = filtered_chr_text__

        #return result_char_ratio, total_sentences, total_words, char_total_count, char_count_, ext_sim_words_key
        return ext_sim_words_key



    #########################################################################
    # 3.유사단어를 문장에서 추출하여 반환한다.
    conflict_sim_words_ratio_result = conflict_sim_words(input_text)




    #########################################################################
    # 4.CONFLICT GRAPH EXPRESSION Analysis  -- 그래프로 그리기
    # conflict(input_text):
    contents = str(input_text)
    token_list_str = text_to_word_sequence(contents) #tokenize
    # 원본문장 단어 중복제거
    token_list_str_set = set(token_list_str)
    print("#"*100)
    print("token_list_str_set:",token_list_str_set)
   
    #print('token_list_str:', token_list_str)
    confict_words_list_basic = ['clash', 'incompatible', 'inconsistent', 'incongruous', 'opposition', 'variance','vary', 'odds', 
                            'differ', 'diverge', 'disagree', 'contrast', 'collide', 'contradictory', 'incompatible', 'conflict',
                            'inconsistent','irreconcilable','incongruous','contrary','opposite','opposing','opposed', 'fight',
                            'antithetical','clashing','discordant','differing','different','divergent','discrepant', 'beef', 'bone to pick',
                            'varying','disagreeing','contrasting','at odds','in opposition','at variance', 'different', 'bone of contention',
                            'battle', 'competition', 'combat', 'rivalry', 'strife', 'struggle', 'war', 'collision',
                            'contention', 'contest', 'emulation', 'encounter', 'engagement', 'fracas', 'fray', 'set-to',
                            'striving', 'tug-of-war', 'conflicted', 'conflicting', 'conflicts', 'disagreement','contrariety',
                            'friction', 'enmity', 'dissension', 'incongruity', 'rancor', 'resistance', 'hostility', 'hatred',
                            'discord', 'debate', 'controversy', 'dispute', 'agitation','matter','matter at hand','problem',
                            'point in question', 'question', 'dispute', 'issue', 'sore point', 'tender spot', 'quarrel', 'discord']

    confict_words_list = confict_words_list_basic + conflict_sim_words_ratio_result #유사단어를 계산결과 반영!
    #중복제거
    confict_words_list_set = set(confict_words_list)
    # print('confict_words_list:', confict_words_list)
    
    # 문장에 들어있는 추출된 conflict 단어들 : count_conflict_list ==================> conflict 단어가 없음(겹치는 단어 없나?)
    count_conflict_list = []
    for ittm in confict_words_list_set:
        for token_list in token_list_str_set:
            if  token_list in ittm : 
                count_conflict_list.append(ittm)
    
    # print()        
    # print('문장에 들어있는 추출된 conflict 단어들:', count_conflict_list)
    # print("#"*100)
    
    count_conflict_list = set(count_conflict_list) #중복제거
    count_conflict_list = list(count_conflict_list) #다시 리스트로 변환
    
    # 전체문장에 들어있는 conflict 단어 수
    nums_conflict_words =  len(count_conflict_list)
    #print('전체문장에 들어있는 conflict 단어 수:', nums_conflict_words)
    
    list_str = contents.split(".")  # 문장별로 분리한다. 분리는 .를 기준으로 한다.   

    listSentiment = []

    sid = SentimentIntensityAnalyzer()

    i=0
    for sentence in tqdm(list_str): #한문장식 가져와서 처리한다.
        ss = sid.polarity_scores(sentence) #긍정, 부정, 중립, 혼합점수 계산
        #print(ss.keys())
        #print('{}: neg:{},neu:{},pos:{},compound:{}'.format(i,ss['neg'],ss['neu'],ss['pos'],ss['compound']))
        #print('{}: neg:{}'.format(i,ss['neg']))
        i +=1
        listSentiment.append([ss['neg'],ss['neu'],ss['pos'],ss['compound']])

    import pandas as pd

    df_sent = pd.DataFrame(listSentiment)
    df_sent.columns = ['neg', 'neu', 'pos','compound']
    reslult_df = df_sent.columns


    df_sent['comp_score'] = df_sent['compound'].apply(lambda c: 'pos' if c >=0  else 'neg')

    df_sent['comp_score'].value_counts()

    conflict_ratio = df_sent['comp_score'].value_counts(normalize=True) #상대적 비율 계산

    # df_sent 의 값은 아래와 같다.

    # neg   neu   pos   compound   comp_score
    # 0   0.000   0.808   0.192   0.2280   pos
    # 1   0.000   1.000   0.000   0.0000   pos
    # 2   0.041   0.778   0.181   0.7269   pos
    # 3   0.044   0.787   0.169   0.6486   pos
    # 4   0.190   0.678   0.132   -0.2144   neg
    # 5   0.000   1.000   0.000   0.0000   pos

    #comp_score를 1 -1 변환
    df_sent.loc[df_sent["comp_score"] == "pos","comp_score"] = 1
    df_sent.loc[df_sent["comp_score"] == "neg","comp_score"] = -1
    

    # df_sent 의 변환된 값은 아래와 같다.
    # neg   neu   pos   compound   comp_score
    # 0   0.000   0.808   0.192   0.2280   1
    # 1   0.000   1.000   0.000   0.0000   1
    # 2   0.041   0.778   0.181   0.7269   1
    # 3   0.044   0.787   0.169   0.6486   1
    # 4   0.190   0.678   0.132   -0.2144   -1
    # 5   0.000   1.000   0.000   0.0000   

    #########################################################################
    # 5. 그래프로 그려보자. 이 코드는 matplotlib 로 그린것임. 종필은 highcharts로 표현할 것
    #########################################################################
    # print("#"*100)
    # print("df_sent['comp_score']:",df_sent['comp_score'])
    # print("idx:",df_sent[df_sent['comp_score'] == 1].index)
    # print("#"*100)
    
    list_pos_neg = []
    list_df_sent = list(df_sent['comp_score']) 
    idx = 0 
    for df in  list_df_sent : 
        
         
        if idx < len(list_df_sent)-1 and df != list_df_sent[idx+1]  and  list_str[idx+1] : 
            
            list_pos_neg.append(list_str[idx+1])
            print("list_pos_neg_"+str(idx+1)+":", list_str[idx+1])
        idx += 1 
        
    
    
    # neg_pos_list = []
    # list_str[idx] 
    
 
    
    
    
    # from matplotlib import pyplot as plt

    # plt.plot(df_sent)
    # plt.xlabel('STORY')
    # plt.ylabel('CONFLICT')
    # plt.title('FLOW ANALYSIS')
    # plt.legend(['neg','neu','pos','compound','reslult'])
    # plt.show()

    #########################################################################
    # 6.ACTION VERB로 그래프 그리기


    #입력한 글을 모두 단어로 쪼개로 리스트로 만들기 - 
    essay_input_corpus_ = str(input_text) #문장입력
    essay_input_corpus_ = essay_input_corpus_.lower()#소문자 변환

    sentences_  = sent_tokenize(essay_input_corpus_) #문장단위로 토큰화(구분)되어 리스에 담김

    # 문장을 토크큰화하여 해당 문장에 Action Verbs가 있는지 분석 부분 코드임 ---> 뒤에서 나옴 아래 777로 표시된 코드부분에서 sentences_ 값 재활용

    split_sentences_ = []
    for sentence in sentences_:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences_.append(words)
        
    # 입력한 문장을 모두 리스트로 변환
    input_text_list = [y for x in split_sentences_ for y in x] # 이중 리스트 Flatten

    #리스로 변환된 값 확인
    #input_text_list 

    #csv 파일에서 Action Verbs 단어 사전 불러오기
    import pandas as pd

    #Awards 데이터 불러오기
   ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### -- > kyle upgrade...
    # after upgrade, return below code!
    #data_action_verbs = pd.read_csv('/var/www/html/essayfit/accu_ps/plot_conflict/actionverbs.csv')
    data_action_verbs = pd.read_csv('./accu_ps/plot_conflict/actionverbs.csv')
    data_ac_verbs_list = data_action_verbs.values.tolist()
    verbs_list = [y for x in data_ac_verbs_list for y in x]

    #########################################################################
    # 7.Action Verbs 유사단어를 추출하여 리스트로 반환

    def actionverb_sim_words(text):

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

        # ACTION VERBS 표현하는 단어들을 리스트에 넣어서 필터로 만들고
        ##################################################
        # verbs_list

        ####문장에 list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
        
        #우선 토큰화한다.
        retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
        token_input_text = retokenize.tokenize(essay_input_corpus)
        #print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
        #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
        filtered_chr_text = []
        for k in token_input_text:
            for j in verbs_list:
                if k == j:
                    filtered_chr_text.append(j)
        
        #print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
        
        filtered_chr_text_ = set(filtered_chr_text) #중복제거
        filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
        #print (filtered_chr_text__) # 중복값 제거 확인
        
        #         for i in filtered_chr_text__:
        #             ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
                
        #         char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 표현 수
        #         char_count_ = len(filtered_chr_text__) #중복제거된  표현 총 수
                    
        #         result_char_ratio = round(char_total_count/total_words * 100, 2)
                
        #         df_conf_words = pd.DataFrame(ext_sim_words_key, columns=['words','values']) #데이터프레임으로 변환
        #         df_r = df_conf_words['words'] #words 컬럼 값 추출
        #         ext_sim_words_key = df_r.values.tolist() # 유사단어 추출

        #return result_char_ratio, total_sentences, total_words, char_total_count, char_count_, ext_sim_words_key
        ext_sim_words_key = filtered_chr_text__
        return ext_sim_words_key


    # 입력문장에서 맥락상 Aciton Verbs와 유사한 의미의 단어를 추출
    ext_action_verbs = actionverb_sim_words(input_text)

    #########################################################################
    # 8.이제 입력문장에서 사용용된 Action Verbs 단어를 비교하여 추출해보자.

    # Action Verbs를 모두 모음(직접적인 단어, 문맥상 유사어 포함)
    all_ac_verbs_list = verbs_list + ext_action_verbs

    #입력한 리스트 값을 하나씩 불러와서 데이터프레이에 있는지 비교 찾아내서 해당 점수를 가져오기
    graph_calculation_list =[0]
    get_words__ = []
    counter= 0
    for h in input_text_list: #데이터프레임에서 인덱스의 값과 비교하여
        if h in all_ac_verbs_list: #df에 특정 단어가 있다면, 해당하는 컬럼의 값을 가져오기
            get_words__.append(h) # 동일하면 저장하기
            #print('counter :', counter)
            graph_calculation_list.append(round(graph_calculation_list[counter]+2,2))
            #print ('graph_calculation_list[counter]:', graph_calculation_list[counter])
            #graph_calculation_list.append(random.randrange(1,10))
            counter += 1
        else: #없다면
            #print('counter :', counter)
            graph_calculation_list.append(round(graph_calculation_list[counter]-0.1,2)) 
            counter += 1
    

            
    #문장에 Action Verbs 추출확인
    #print('Action Verbs:', get_words__)
    nums_action_verbs = len(get_words__)
    #print('Number of Action Verbs:', nums_action_verbs)


    def divide_list(l, n): 
        # 리스트 l의 길이가 n이면 계속 반복
        for i in range(0, int(len(l)), int(n)): 
            yield l[i:i + int(n)] 
        
    # 한 리스트에 몇개씩 담을지 결정 = 20개씩

    n = len(graph_calculation_list)/20

    result_gr = list(divide_list(graph_calculation_list, n))

    gr_cal = []
    for regr in result_gr:
        avg_gr = sum(regr,0.0)/len(regr) #묶어서 평균을 내고 
        gr_cal.append(abs(round(avg_gr,2))) #절대값을 전환해서


    graph_calculation_list = gr_cal  ## 그래프를 위한 최종결과 계산 후, 이것을 딕셔너리로 반환하여 > 그래프로 표현하기
    #########################################################################
    # 9. 그래프 출력 : 문장 전체를 단어로 분리하고, Action verbs가 사용된 부분을 그래프로 표시

    # 전체 글에서 Action verbs가 언급된 부분을 리스트로 계산
    # graph_calculation_list 

    #그래프로 표시됨
    # plt.plot(graph_calculation_list)
    # plt.xlabel('STORY')
    # plt.ylabel('ACTON VERBS')
    # plt.title('USAGE OF ACTION VERBS ANALYSIS')
    # plt.legend(['action verbs'])
    # plt.show()

    #########################################################################
    # 10.입력한 에세이 문장에서 Action Verbs가 얼마나 포함되어 있는지 포함비율 분석
    action_verbs_ratio = round(len(get_words__)/len(input_text_list) *100, 3)

    #print ("ACTION VERBS RATIO :", action_verbs_ratio )


    #########################################################################
    # 11. 글속에 감정이 얼마나 표현되어 있는지 분석 - origin (Bert pre trained model 활용)
    from pprint import pprint

    # 다운로드 했기 때문에 실행시는 사용하지 않음
    # tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
    # model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

    goemotions = MultiLabelPipeline(
        model=model,
        tokenizer=tokenizer,
        threshold=0.3
    )
    #결과확인
    #print(goemotions(texts))
    ########## 여기서는 최초 입력 에세이를 적용한다. input_text !!!!!!!!
    re_text = input_text.split(".")

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


    # 중립적인 감정을 제외하고, 입력한 문장에서 다양한 감정을 모두 추출하고 어떤 감정이 있는지 계산해보자
    unique = []
    for r in flat_list:
        if r == 'neutral':
            pass
        else:
            unique.append(r)
            
    


    ## 감정기복 분석 --> plot & conflict 의 문장 생성부분
    # neutural감정 제외하고 감정추출하여 계산하자
    def ext_shift_emoiton_ratio(unique):
        for u_itm in unique:
            #shift_counter +1 -1로 값을 positive, negative로 계산해서 절대값이 적게 나올수록 감정기복이 심하겠지. 많으면 기복이 적음 
            shift_counter = 0
            positive_emo_group_list = ['admiration', 'amusement', 'approval', 'caring', 'curiosity', 'desire', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'realization', 'relief', 'surprise']
            negative_emo_group_list = ['anger', 'annoyance', 'confusion', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness']
            #positive 감정에 속하면
            if u_itm in positive_emo_group_list:
                shift_counter += 1
            else: #negative일 경우 -1
                shift_counter -= 1
        ##############################################################
        ##############################################################
        re_shift_cnt = abs(shift_counter)#절대값
        if re_shift_cnt > 10: #감정기복이 어느 한쪽으로 편향이 6번이상 된다면 >>>>>>>>>>>>>>> !! 기준값 결정하는것 중요(group 평균값 계산해서 적용할 것)
            shift_emo_ratio = 'sparse'
        elif re_shift_cnt <= 9 and re_shift_cnt > 5:
            shift_emo_ratio = 'moderate'
        else: #5이하라면 감정기복이 심함 pos neg가 많이교차됨(반반일 경우)
            shift_emo_ratio = 'frequent'
        ##############################################################
        ##############################################################

        # 결과해석
        # shift_emo_ratio : 'sparse', 'moderate', 'frequent' 중 1개 출력됨
        # re_shift_cnt : 숫자 하나가 출력됨, 이 값은 변동폭을 결정하는 값임. 작을수록 변동폭이 크다는 의미임

        return shift_emo_ratio, re_shift_cnt
    
    
    
    ## 감정분석한 원본 문장 추출하여 웹페이지에 출력(editer 기능 구현 부분)
    

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
    #result_emo_swings
    # print("문장에 표현된 감정 비율 : ", result_emo_swings)
    # print("====================================================================")


    #########################################################################
    # 12. SETTING RATIO 계산

    def setting_anaysis(text):

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

        #setting을 표현하는 단어들을 리스트에 넣어서 필터로 만들고
        location_list = ['above', 'behind','below','beside','betweed','by','in','inside','near',
                        'on','over','through']
        time_list = ['after', 'before','by','during','from','on','past','since','through','to','until','upon']
        
        movement_list = ['against','along','down','from','into','off','on','onto','out of','toward','up','upon']
        
        palce_terrain_type_list = ['wood', 'forest', 'copse', 'bush', 'trees', 'stand',
                                    'swamp', 'marsh', 'wetland', 'fen', 'bog', 'moor', 'heath', 'fells', 'morass',
                                    'jungle', 'rainforest', 'cloud forest','plains', 'fields', 'grass', 'grassland', 
                                    'savannah', 'flood plain', 'flats', 'prairie','tundra', 'iceberg', 'glacier', 
                                    'snowfields','hills', 'highland,' 'heights', 'plateau', 'badland', 'kame', 'shield',
                                    'downs', 'downland', 'ridge', 'ridgeline','hollow,' 'valley',' vale','glen', 'dell',
                                    'mountain', 'peak', 'summit', 'rise', 'pass', 'notch', 'crown', 'mount', 'switchback',
                                    'furth','canyon', 'cliff', 'bluff,' 'ravine', 'gully', 'gulch', 'gorge',
                                    'desert', 'scrub', 'waste', 'wasteland', 'sands', 'dunes',
                                    'volcano', 'crater', 'cone', 'geyser', 'lava fields']
        
        water_list = ['ocean', 'sea', 'coast', 'beach', 'shore', 'strand','bay', 'port', 'harbour', 'fjord', 'vike',
                    'cove', 'shoals', 'lagoon', 'firth', 'bight', 'sound', 'strait', 'gulf', 'inlet', 'loch', 
                    'bayou','dock', 'pier', 'anchorage', 'jetty', 'wharf', 'marina', 'landing', 'mooring', 'berth', 
                    'quay', 'staith','river', 'stream', 'creek', 'brook', 'waterway', 'rill','delta', 'bank', 'runoff',
                    'bend', 'meander', 'backwater','lake', 'pool', 'pond', 'dugout', 'fountain', 'spring', 
                    'watering-hole', 'oasis','well', 'cistern', 'reservoir','waterfall', 'falls', 'rapids', 'cataract', 
                    'cascade','bridge', 'crossing', 'causeway', 'viaduct', 'aquaduct', 'ford', 'ferry','dam', 'dike', 
                    'bar', 'canal', 'ditch','peninsula', 'isthmus', 'island', 'isle', 'sandbar', 'reef', 'atoll', 
                    'archipelago', 'cay','shipwreck', 'derelict']
        
        
        outdoor_places_list = ['clearing', 'meadow', 'grove', 'glade', 'fairy ring','earldom', 'fief', 'shire',
                                'ruin', 'acropolis', 'desolation', 'remnant', 'remains',
                                'henge', 'cairn', 'circle', 'mound', 'barrow', 'earthworks', 'petroglyphs',
                                'lookout', 'aerie', 'promontory', 'outcropping', 'ledge', 'overhang', 'mesa', 'butte',
                                'outland', 'outback', 'territory', 'reaches', 'wild', 'wilderness', 'expanse',
                                'view', 'vista', 'tableau', 'spectacle', 'landscape', 'seascape', 'aurora', 'landmark',
                                'battlefield', 'trenches', 'gambit', 'folly', 'conquest', 'claim', 'muster', 'post',
                                'path', 'road', 'track', 'route', 'highway', 'way', 'trail', 'lane', 'thoroughfare', 'pike',
                                'alley', 'street', 'avenue', 'boulevard', 'promenade', 'esplande', 'boardwalk',
                                'crossroad', 'junction', 'intersection', 'turn', 'corner','plaza', 'terrace', 'square', 
                                'courtyard', 'court', 'park', 'marketplace', 'bazaar', 'fairground','realm', 'land', 'country',
                                'nation', 'state', 'protectorate', 'empire', 'kingdom', 'principality','domain', 'dominion',
                                'demesne', 'province', 'county', 'duchy', 'barony', 'baronetcy', 'march', 'canton']

        
        underground_list = ['pit', 'hole', 'abyss', 'sinkhole', 'crack', 'chasm', 'scar', 'rift', 'trench', 'fissure',
                            'cavern', 'cave', 'gallery', 'grotto', 'karst',
                            'mine', 'quarry', 'shaft', 'vein','graveyard', 'cemetery',
                            'darkness', 'shadow', 'depths', 'void','maze', 'labyrinth',
                            'tomb', 'grave', 'crypt', 'sepulchre', 'mausoleum', 'ossuary', 'boneyard']
                            
        living_places_list = ['nest', 'burrow', 'lair', 'den', 'bolt-hole', 'warren', 'roost', 'rookery', 'hibernaculum',
                            'home', 'rest', 'hideout', 'hideaway', 'retreat', 'resting-place', 'safehouse', 'sanctuary',
                            'respite', 'lodge','slum', 'shantytown', 'ghetto','camp', 'meeting place,' 'bivouac', 'campsite', 
                            'encampment','tepee', 'tent', 'wigwam', 'shelter', 'lean-to', 'yurt','house', 'mansion', 'estate',
                            'villa','hut', 'palace', 'outbuilding', 'shack tenement', 'hovel', 'manse', 'manor', 'longhouse',
                            'cottage', 'cabin','parsonage', 'rectory', 'vicarge', 'friary', 'priory','abbey', 'monastery', 
                            'nunnery', 'cloister', 'convent', 'hermitage','castle', 'keep', 'fort', 'fortress', 'citadel', 
                            'bailey', 'motte', 'stronghold', 'hold', 'chateau', 'outpost', 'redoubt',
                            'town', 'village', 'hamlet', 'city', 'metropolis','settlement', 'commune']

        building_facilities_list = ['temple', 'shrine', 'church', 'cathedral', 'tabernacle', 'ark', 'sanctum', 'parish', 'university',
                                    'chapel', 'synagogue', 'mosque','pyramid', 'ziggurat', 'prison', 'jail', 'dungeon',
                                    'oubliette', 'hospital', 'hospice', 'stocks', 'gallows','asylum', 'madhouse', 'bedlam',
                                    'vault', 'treasury', 'warehouse', 'cellar', 'relicry', 'repository',
                                    'barracks', 'armoury','sewer', 'gutter', 'catacombs', 'dump', 'middens', 'pipes', 'baths', 'heap',
                                    'mill', 'windmill', 'sawmill', 'smithy', 'forge', 'workshop', 'brickyard', 'shipyard', 'forgeworks',
                                    'foundry','bakery', 'brewery', 'almshouse', 'counting house', 'courthouse', 'apothecary', 'haberdashery', 'cobbler',
                                    'garden', 'menagerie', 'zoo', 'aquarium', 'terrarium', 'conservatory', 'lawn', 'greenhouse',
                                    'farm', 'orchard', 'vineyard', 'ranch', 'apiary', 'farmstead', 'homestead',
                                    'pasture', 'commons', 'granary', 'silo', 'crop','barn', 'stable', 'pen', 'kennel', 'mews', 'hutch', 
                                    'pound', 'coop', 'stockade', 'yard', 'lumber yard','tavern', 'inn', 'pub', 'brothel', 'whorehouse',
                                    'cathouse', 'discotheque','lighthouse', 'beacon','amphitheatre', 'colosseum', 'stadium', 'arena', 
                                    'circus','academy', 'university', 'campus', 'college', 'library', 'scriptorium', 'laboratory', 
                                    'observatory', 'museum']
        
        
        architecture_list = ['hall', 'chamber', 'room','nave', 'aisle', 'vestibule',
                            'antechamber', 'chantry', 'pulpit','dome', 'arch', 'colonnade',
                            'stair', 'ladder', 'climb', 'ramp', 'steps',
                            'portal', 'mouth', 'opening', 'door', 'gate', 'entrance', 'maw',
                            'tunnel', 'passage', 'corridor', 'hallway', 'chute', 'slide', 'tube', 'trapdoor',
                            'tower', 'turret', 'belfry','wall', 'fortifications', 'ramparts', 'pallisade', 'battlements',
                            'portcullis', 'barbican','throne room', 'ballroom','roof', 'rooftops', 'chimney', 'attic',
                            'loft', 'gable', 'eaves', 'belvedere','balcony', 'balustrade', 'parapet', 'walkway', 'catwalk',
                            'pavillion', 'pagoda', 'gazebo','mirror', 'glass', 'mere','throne', 'seat', 'dais',
                            'pillar', 'column', 'stone', 'spike', 'rock', 'megalith', 'menhir', 'dolmen', 'obelisk',
                            'statue', 'giant', 'head', 'arm', 'leg', 'body', 'chest', 'body', 'face', 'visage', 'gargoyle', 'grotesque',
                            'fire', 'flame', 'bonfire', 'hearth', 'fireplace', 'furnace', 'stove','window', 'grate', 'peephole', 
                            'arrowslit', 'slit', 'balistraria', 'lancet', 'aperture', 'dormerl']
        
        
        setting_words_filter_list = location_list + time_list + movement_list + palce_terrain_type_list + water_list + outdoor_places_list + underground_list + underground_list + living_places_list + building_facilities_list + architecture_list

        
        ####문장에 setting_words_filter_list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
        #우선 토큰화한다.
        retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
        token_input_text = retokenize.tokenize(essay_input_corpus)
        #print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
        #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
        filtered_setting_text = []
        for k in token_input_text:
            for j in setting_words_filter_list:
                if k == j:
                    filtered_setting_text.append(j)
        
        #print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
        
        filtered_setting_text_ = set(filtered_setting_text) #중복제거
        filtered_setting_text__ = list(filtered_setting_text_) #다시 리스트로 변환
        #print (filtered_setting_text__) # 중복값 제거 확인
        
    #         for i in filtered_setting_text__:
    #             ext_setting_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
        setting_total_count = len(filtered_setting_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현 수
        setting_count_ = len(filtered_setting_text__) #중복제거된 setting표현 총 수
            
        result_setting_words_ratio = round(setting_total_count/total_words * 100, 2)
        #return result_setting_words_ratio, total_sentences, total_words, setting_total_count, setting_count_, ext_setting_sim_words_key
        return result_setting_words_ratio


    # 셋팅 비율 계산
    settig_ratio_re = setting_anaysis(input_text)
    #print("====================================================================")
    #print("SETTING RATIO : ", settig_ratio_re)
    #print("====================================================================")


    ###################################################################################
    # 13. PLOT COMPLEXITY 계산¶ - 캐릭터 20% + conflict 40% + 감정기복 30% + setting 10%
    ###################################################################################

    def character(text):

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
                        'twin','uncle','widow','widower','wife','ex-wife']
        
        ####문장에 char_list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
        #우선 토큰화한다.
        retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
        token_input_text = retokenize.tokenize(essay_input_corpus)
        #print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
        #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
        filtered_chr_text = []
        for k in token_input_text:
            for j in character_list:
                if k == j:
                    filtered_chr_text.append(j)
        
        #print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
        
        filtered_chr_text_ = set(filtered_chr_text) #중복제거
        filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
        #print (filtered_chr_text__) # 중복값 제거 확인
        
        for i in filtered_chr_text__:
            ext_sim_words_key = model.wv.most_similar_cosmul(i) #모델적용
        
        char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 캐릭터 표현 수
        char_count_ = len(filtered_chr_text__) #중복제거된 캐릭터 표현 총 수
            
        result_char_ratio = round(char_total_count/total_words * 100, 2)

        #return result_char_ratio, total_sentences, total_words, char_total_count, char_count_, ext_sim_words_key
        return result_char_ratio


    #######################################################################
    ##########################   Plot complexity  ######################
    #######################################################################
    # 이제 최종 계산을 해보자.
    # character_ratio_result #캐릭터 비율 20%
    # result_emo_swings # 감정기복 비율 30%
    # conflict_word_ratio #CONFLICT 비율 계산 40%
    # settig_ratio_re #Setting 비율 계산 10%
    # 전체 문장에서 캐릭터를 의미하는 단어나 유사어 비율 

    character_ratio_result = character(input_text)
    character_ratio_result
    #print("전체 문장에서 캐릭터를 의미하는 단어나 유사어 비율 :", character_ratio_result)

    ###########################################################
    ############# Degree of Conflict  비율 계산 #################
    conflict_word_ratio = round(len(count_conflict_list) / len(input_text_list) * 1000, 1)  
    #print("Degree of conflict  단어가 전체 문장(단어)에서 차지하는 비율 계산 :", conflict_word_ratio)

    global coflict_ratio
    coflict_ratio = [conflict_word_ratio] #그래프로 표현하는 값



    ###########################################################
    ############# Emotional Rollercoaster  비율 계산 #################
    #print("감정기복비율 :", result_emo_swings) 

    # 셋팅비율 계산
    #print("셋팅비율 계산 : ", settig_ratio_re)

    # 4개의 값을 리스트로 담는다.
    de_flt_list = [character_ratio_result, result_emo_swings, conflict_word_ratio, settig_ratio_re]


    import numpy
    numpy.mean(de_flt_list) #평균
    numpy.var(de_flt_list) #분산
    numpy.std(de_flt_list) #표준편차

    #######################################################################
    ############# Plot complexity  st_input 표준편차 비율 계산 #################
    de_flt_list_ = [character_ratio_result*2, result_emo_swings*3, conflict_word_ratio*4, settig_ratio_re]
    numpy.mean(de_flt_list_) # 평균
    numpy.var(de_flt_list_) # 분산
    st_input = numpy.std(de_flt_list_) # 표준편차 ----> 이 값으로 계산
    #print("Plot Complxity :", st_input )

    global plot_comp_ratio

    plot_comp_ratio = [round(st_input, 2)]
    
    
    ## 감정기복 분석 --> plot & conflict 의 문장 생성부분으로 (결정된 값 frequent, moderate, sparse 중 하나, 숫자)숫자가 작으면 변통폭이 큼
    ext_shift_emoiton_re = ext_shift_emoiton_ratio(unique)

    #print("===============================================================================")
    #print("======================      Degree of Conflict   ==============================")
    #print("===============================================================================")

    
    ### return 값 설명 ###
    # 0.st_input  : plot complexity
    # 1.result_emo_swings : emotion rollercoster
    # 2.conflict_word_ratio : degree of conflict
    # 3.df_sent : 
    # 4.graph_calculation_list : raph_calculation_list
    # 5.count_conflict_list : 컨플릭 단어들 리스트
    # 6.nums_conflict_words : conflict words number
    # 7.get_words__ : Action Verbs
    # 8.nums_action_verbs : Action verbs 수
    # 9.ext_shift_emoiton_re : Shifts Between Positive and Negative Sentiments의 감정기복 값 계산 결과로 문장생성 frequent / moderate / sparse
    
    return st_input, result_emo_swings, conflict_word_ratio, df_sent, graph_calculation_list, count_conflict_list, nums_conflict_words, get_words__, nums_action_verbs, ext_shift_emoiton_re,list_pos_neg




def ai_plot_coflict_total_analysis(input_text):

    plot_conf_re = ai_plot_conf(input_text)
    
    count_conflict_list_re = plot_conf_re[5] # conflict words list
    #nums_conflict_words_re = plot_conf_re[6] # conflict words number
    #get_words__re = plot_conf_re[7] # Action Verbs list
    #nums_action_verbs_re = plot_conf_re[8] # Action verbs number
    
    #Shifts Between Positive and Negative Sentiments의 감정기복 값 계산 결과로 문장생성 frequent / moderate / sparse , 숫자값
    shifts_btw_neg_pos = plot_conf_re[9]
    
    #print("1명의 에세이 결과 계산점수 :", plot_conf_re)
    #1명의 에세이 결과 계산점수 : (28.602484157848945, 25.0, 0.3,df_sent)

    # 위에서 계산한 총 4개의 값을 개인, 그룹의 값과 비교하여 lacking, ideal, overboard 계산
    
    # 개인에세이 값 계산 4가지 결과 추출 >>> personal_value 로 입력됨
    plot_complexity = plot_conf_re[0]
    emotional_rollercoaster = plot_conf_re[1]
    degree_conflict = plot_conf_re[2]
    
    graph_calculation_list = plot_conf_re[4]
    
    ########################################################
    ## 1000명 데이터의 각 값(char_desc_mean)의 평균 값 전달.>>> 고정값으로 미리 계산하여 입력 ai_plot_coflict_1000data_preprocessing 코드 참조
    plot_conflict_all_mean = [80, 64, 0.314]
    group_db_fin_result_plot = [5.0]
    ########################################################

    plot_complexity_mean = plot_conflict_all_mean[0] #첫번째 값을 가져옴
    emotional_rollercoaster_mean = plot_conflict_all_mean[1] #
    degree_conflict_mean = plot_conflict_all_mean[2] #


    def lackigIdealOverboard(group_mean, personal_value): # group_mean: 1000명 평균, personal_value|:개인값
        ideal_mean = group_mean
        one_ps_char_desc = personal_value
        #최대, 최소값 기준으로 구간설정. 구간비율 30% => 0.3으로 설정
        min_ = int(ideal_mean-ideal_mean*0.6)
        #print('min_', min_)
        max_ = int(ideal_mean+ideal_mean*0.6)
        #print('max_: ', max_)
        div_ = int(((ideal_mean+ideal_mean*0.6)-(ideal_mean-ideal_mean*0.6))/3)
        #print('div_:', div_)

        #결과 판단 Lacking, Ideal, Overboard
        cal_abs = abs(ideal_mean - one_ps_char_desc) # 개인 - 단체 값의 절대값계산

        #print('cal_abs 절대값 :', cal_abs)
        compare7 = (one_ps_char_desc + ideal_mean)/6
        compare6 = (one_ps_char_desc + ideal_mean)/5
        compare5 = (one_ps_char_desc + ideal_mean)/4
        compare4 = (one_ps_char_desc + ideal_mean)/3
        compare3 = (one_ps_char_desc + ideal_mean)/2
        #print('compare7 :', compare7)
        #print('compare6 :', compare6)
        #print('compare5 :', compare5)
        #print('compare4 :', compare4)
        #print('compare3 :', compare3)



        if one_ps_char_desc > ideal_mean: # 개인점수가 평균보다 클 경우는 overboard
            if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
                #print("Overboard: 2")
                result = 2 #overboard
                score = 1
            elif cal_abs > compare4: # 28
                #print("Overvoard: 2")
                result = 2
                score = 2
            elif cal_abs > compare5: # 22
                #print("Overvoard: 2")
                result = 2
                score = 3
            elif cal_abs > compare6: # 18
                #print("Overvoard: 2")
                result = 2
                score = 4
            else:
                #print("Ideal: 1")
                result = 1
                score = 5
        elif one_ps_char_desc < ideal_mean: # 개인점수가 평균보다 작을 경우 lacking
            if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
                #print("Lacking: 2")
                result = 0
                score = 1
            elif cal_abs > compare4: # 28
                #print("Lacking: 2")
                result = 0
                score = 2
            elif cal_abs > compare5: # 22
                #print("Lacking: 2")
                result = 0
                score = 3
            elif cal_abs > compare6: # 18
                #print("Lacking: 2")
                result = 0
                score = 4
            else:
                print("Ideal: 1")
                result = 1
                score = 5
                
        else:
            #print("Ideal: 1")
            result = 1
            score = 5

        return result, score


    plot_complexity_result = lackigIdealOverboard(plot_complexity_mean, plot_complexity)
    emotional_rollercoaster_result = lackigIdealOverboard(emotional_rollercoaster_mean, emotional_rollercoaster)
    degree_conflict_result = lackigIdealOverboard(degree_conflict_mean, degree_conflict)

    fin_result = [plot_complexity_result, emotional_rollercoaster_result, degree_conflict_result]
    #print("fin_result:", fin_result)  # [(0:lacking, 1:score), (0:lacking, 2:score), (2:overboard, 1:score)]

    each_fin_result = [fin_result[0][0], fin_result[1][0], fin_result[2][0]]

    # 최종 character  전체 점수 계산
    overall_character_rating = [round((fin_result[0][1]+ fin_result[1][1] + fin_result[2][1])/3,2)]

    result_final = each_fin_result + overall_character_rating + group_db_fin_result_plot + coflict_ratio + plot_comp_ratio

    df_sent = plot_conf_re[3]
    
    neg =  list(map(float, df_sent["neg"]))
    neu =  list(map(float, df_sent["neu"]))
    pos =  list(map(float, df_sent["pos"]))
    compound =  list(map(float, df_sent["compound"]))
    
    #print(df_sent)
    
    #print("neg>>>>>",neg)
    #print("neu>>>>>",neu)
    #print("pos>>>>>",pos)
    #print("componud>>>",compound)
    
    #print ( "graph_calculation_list" , graph_calculation_list) 

    data = {
        
            "result_all_plot":result_final[3], 

            "emotional_rollercoaster" : round(emotional_rollercoaster,2), 
            "plot_complexity" : round(plot_complexity,2), 
            "degree_conflict" : round(degree_conflict,2), 
            
            "result_plot_complexity" : result_final[0],
            "result_emotional_rollercoaster": result_final[1],
            "result_degree_conflict" : result_final[2],
            
            "neg" : neg,
            "neu" : neu,
            "pos" : pos,
            "compound" : compound,
            "graph_calculation_list" : graph_calculation_list,
            "Shifts Between Positive and Negative Sentiments" : shifts_btw_neg_pos,
            "conflict words list" : count_conflict_list_re, # 문장에서 추출한 conflict 단어 모음
            "list_pos_neg" : plot_conf_re[10]
            

        }
    
    return data 



# select prompt number to get intended mood
def intended_mood_by_prompt(promptNo):

    intended_mood = []
    if promptNo == 'ques_one':
        intended_mood = ['joy', 'pride', 'approval']
    elif promptNo == "ques_two":
        intended_mood = ['disappointment', 'fear', 'confusion']
    elif promptNo == "ques_three":
        intended_mood = ['curiosity', 'disapproval', 'realization']
    elif promptNo == "ques_four":
        intended_mood = ['gratitude', 'surprise', 'admiration']
    elif promptNo == "ques_five":
        intended_mood = ['realization', 'pride', 'admiration']
    elif promptNo == "ques_six":
        intended_mood = ['curiosity', 'excitement', 'confusion']
    elif promptNo == "ques_seven":
        intended_mood = ['joy', 'approval','disappointment', 'fear', 
                         'confusion', 'disapproval', 'realization',
                        'gratitude', 'surprise', 'admiration', 'pride',
                        'curiosity', 'excitement', ]
    else:
        pass
    
    return intended_mood



# 에세이의 감성분석, 입력값(essay, selected prompt number)
def ai_emotion_analysis(input_text, promt_number):
    # . 로 구분하여 리스트로 변환
    re_text = input_text.split(".")
    #print("re_text type: ", type(re_text))
        
    texts = cleaning(re_text)
    re_emot =  goemotions(texts)
    df = pd.DataFrame(re_emot)
    #print("dataframe:", df)
    label_cnt = df.count()
    #print(label_cnt)
 
    #추출된 감성중 레이블만 다시 추출하고, 이것을 리스트로 변환 후, 이중리스트 flatten하고, 가장 많이 추출된 대표감성을 카운트하여 산출한다.
    result_emotion = list(df['labels'])
    #이중리스트 flatten
    all_emo_types = sum(result_emotion, [])
    #대표감성 추출 : 리스트 항목 카운트하여 가장 높은 값 산출
    ext_emotion = {}
    for i in all_emo_types:
        if i == 'neutral': # neutral 감정은 제거함
            pass
        else:
            try: ext_emotion[i] += 1
            except: ext_emotion[i]=1    
    #print(ext_emotion)
    #결과값 오름차순 정렬 : 추출된 감성 결과가 높은 순서대로 정려하기
    key_emo = sorted(ext_emotion.items(), key=lambda x: x[1], reverse=True)
    #print("Key extract emoitons: ", key_emo)
    
    #가장 많이 추출된 감성 1개
    #key_emo[0]
    
    #가장 많이 추출된 감성 3개
    #key_emo[:2]
    
    #가장 많이 추출된 감성 5개
    key_emo[:5]
    
    result_emo_list = [*sum(zip(re_text, result_emotion),())]
    
    # 결과해석
    # result_emo_list >>> 문장, 분석감성
    # key_emo[0] >>> 가장 많이 추출된 감성 1개로 이것이 에세이이 포함된 대표감성
    # key_emo[:2] 가장 많이 추출된 감성 3개
    # key_emo[:5] 가장 많이 추출된 감성 5개
    top5Emo = key_emo[:5]
    #print('top5Emo : ', top5Emo)
    top5Emotions = [] # ['approval', 'realization', 'admiration', 'excitement', 'amusement']

    try  : 
       
        top5Emotions.append(top5Emo[0][0])
        top5Emotions.append(top5Emo[1][0])
        top5Emotions.append(top5Emo[2][0])
        top5Emotions.append(top5Emo[3][0])
        top5Emotions.append(top5Emo[4][0])

    except : 
        pass 
    
    # 감성추출결과 분류항목 - Intended Mood 별 연관 sentiment
    disturbed =['anger', 'annoyance', 'disapproval', 'confusion', 'disappointment', 'disgust', 'anger']
    suspenseful = ['fear', 'nervousness', 'confusion', 'surprise', 'excitement']
    sad = ['disappointment', 'embarrassment', 'grief', 'remorse', 'sadness']
    joyful = ['admiration', 'amusement', 'excitement', 'joy', 'optimism']
    calm = ['caring', 'gratitude', 'realization', 'curiosity', 'admiration', 'neutral']
    
    re_mood ='' 
    for each_emo in top5Emotions:
        if each_emo in disturbed:
            re_mood = "disturbed"
        elif each_emo in suspenseful:
            re_mood = "suspenseful"
        elif each_emo in sad:
            re_mood = "sad"
        elif each_emo in joyful:
            re_mood ="joyful"
        elif each_emo in calm:
            re_mood ="calm"
        else:
            pass
        
    #입력한 에세이에서 추출한 mood의 str을 리스트로 변환    
    detected_mood = [] #결과값으로 이것을 return할 거임
    detected_mood.append(re_mood)
    
    # intended mood, prompt에서 선택한 내용대로 관련 mood 를 추출
    get_intended_mood = intended_mood_by_prompt(promt_number) # ex) ['disappointment', 'fear', 'confusion']
    
    
    #1, 2nd Senctece 생성
    if re_mood == 'disturbed':
        sentence1 = 'You’ve intended to write the essay in a disturbed mood.'
        sentence2 = 'The AI’s analysis shows that your personal statement’s mood seems to be disturbed.'

    elif re_mood == 'suspenseful':
        sentence1 = 'You’ve intended to write the essay in a suspenseful mood.'
        sentence2 = 'The AI’s analysis shows that your personal statement’s mood seems to be suspenseful.'

    elif re_mood == 'sad':
        sentence1 = 'You’ve intended to write the essay in a sad mood.'
        sentence2 = 'The AI’s analysis shows that your personal statement’s mood seems to be sad.'

    elif re_mood == 'joyful':
        sentence1 = 'You’ve intended to write the essay in a joyful mood.'
        sentence2 = 'The AI’s analysis shows that your personal statement’s mood seems to be joyful.'

    elif re_mood == 'calm':
        sentence1 = 'You’ve intended to write the essay in a calm mood.'
        sentence2 = 'The AI’s analysis shows that your personal statement’s mood seems to be calm.'

    else:
        pass

                    
    # intended mood vs. your essay mood
    intendedMoodByPmt = []
    for each_mood in get_intended_mood: # prompt에서 추출된 mood를 하나씩 가져와서 에세이에서 추출된 mood와 비교
        if each_mood in disturbed:
            intendedMoodByPmt.append(each_mood) 
        elif each_mood in suspenseful:
            intendedMoodByPmt.append(each_mood)
        elif each_mood in sad:
            intendedMoodByPmt.append(each_mood)
        elif each_mood in joyful:
            intendedMoodByPmt.append(each_mood)
        elif each_mood in calm:
            intendedMoodByPmt.append(each_mood)
            
    # 비교하여 3rd Sentece 생성 
    if intendedMoodByPmt == detected_mood: # 두 개의 mood에 해당하는 리스트의 값이 같으면
        sentence3 = """It seems that the mood portrayed in your essay is coherent with what you've intended!"""
    elif intendedMoodByPmt == ['disturbed']: # 같지 않다면 다음 항목을 각각 비교
        sentence3 = """If you wish to shift the essay’s direction towards your original intention, you may consider including more conflicts and how you’ve struggled to resolve them."""
    elif intendedMoodByPmt == ['suspenseful']:
        sentence3 = """If you wish to shift the essay’s direction towards your original intention, you may consider including more incidents, actions, and dynamic elements."""
    elif intendedMoodByPmt == ['sad']:
        sentence3 = """If you wish to shift the essay’s direction towards your original intention, you may consider including more sympathetic stories about difficult times in life."""
    elif intendedMoodByPmt == ['joy']:
        sentence3 = """If you wish to shift the essay’s direction towards your original intention, you may consider including more lighthearted life stories and the positive lessons you draw from them."""
    elif intendedMoodByPmt == ['calm']:
        sentence3 = """If you wish to shift the essay’s direction towards your original intention, you may consider including more self-reflection, intellectual topics, or observations that shaped you."""
    else:
        sentence3 = """ Try Again! """
        
    #################################################################################       
    #1000 합격한 에세이의 평균 Top 5 sentiment
    #결과는 very close / somewhat close / weak 으로 나와야함
    # 각 값은 1000명의 평균에세이값을 산출하여 적용해야함, 지금 값은 dummmy values
    prompt_1_sent_mean = [('joy', 8), ('approval', 5), ('disappointment',6),('confusion',7),('gratitude',7)] 
    prompt_2_sent_mean = [('disappointment',6),('confusion',7),('joy', 8), ('approval', 5), ('disappointment',6)]
    prompt_3_sent_mean = [('curiosity',7),('disapproval',6),('disappointment',6),('confusion',7),('gratitude',7)]
    prompt_4_sent_mean = [('gratitude',8),('surprise',6),('disappointment',6),('confusion',7),('gratitude',7)]
    prompt_5_sent_mean = [('realization',5),('admiration',4),('disappointment',6),('confusion',7),('gratitude',7)]
    prompt_6_sent_mean = [('excitement',9),('confusion',5),('disappointment',6),('confusion',7),('gratitude',7)]
    prompt_7_sent_mean = [('gratitude',7),('joy',5),('disappointment',6),('confusion',7),('gratitude',7)]
    #################################################################################
    accepted_essay_av_value = []
    
    if promt_number == 'ques_one': # 1번 문항을 선택했을 경우(문항선택 'prompt_1 ~ 7')
        accepted_essay_av_value = prompt_1_sent_mean
        
    elif promt_number == 'ques_two':
        accepted_essay_av_value = prompt_2_sent_mean
        
    elif promt_number == 'ques_three':
        accepted_essay_av_value = prompt_3_sent_mean
        
    elif promt_number == 'ques_four':
        accepted_essay_av_value = prompt_4_sent_mean
        
    elif promt_number == 'ques_five':
        accepted_essay_av_value = accepted_essay_av_value = prompt_5_sent_mean
        
    elif promt_number == 'ques_six':
        accepted_essay_av_value = prompt_6_sent_mean
        
    elif promt_number == 'ques_seven':
        accepted_essay_av_value = prompt_7_sent_mean
    else:
        pass
    
    
    
    # 결과해석

    # result_emo_list: 문장 + 감성분석결과
    # intendedMoodByPmt : intended mood 
    # detected_mood : 대표 Mood
    # sentence1,sentence2, sentence3 : intended mood vs. your mood 비교결과에 대한 문장생성 커멘트 
    
    # 대표감성 5개 추출(학생 1명거임) : key_emo[:5]
    # 합격한 한생의 prompt별 대표감성 2개(1000명 평균) : accepted_essay_av_value
    
    # In-depth Sentiment Analysis 매칭되는 결과에따라서 very close / somewhat close / weak 결정
    ps_ext_emo =[] # 개인 에세이에서 추출한 5개의 대표감성
    for itm in key_emo[:5]:
        #print(itm[0])
        ps_ext_emo.append(itm[0])

    #print(ps_ext_emo)
    
    group_ext_emo = [] # 그룹 에세이에서 추출한 5개의 평균 대표감성 5개
    for item_2 in accepted_essay_av_value:
        group_ext_emo.append(item_2[0])
    
    #print(group_ext_emo)
    
    #두 값을 비교하여 very close / somewhat close / weak 결정
    #중복요소를 추출하여 카운팅하면 두 총 리스트의 값 중에서 중복요소가 몇개 있는지 알 수 있을때 유사도를 계산할 수 있음
    count={}
    sum_emo = ps_ext_emo + group_ext_emo
    for m in sum_emo:
        try: count[m] += 1
        except: count[m] = 1
    #print('중복값:', count)
    
    compare_re = []
    for value in count.values(): # 딕셔너리의 벨류 값을 하나씩 가져와서 
        if value > 1: # 1보다 큰 수는 중복된 수 이기 때문에 
            compare_re.append(value) # 중복된 수를 새로운 리스트 compare_re에 넣고
        else:
            pass
        
    sum_compare_re = sum(compare_re) 
    # 리스트의 숫자를 모두 더해서 최종 비교를 할거임,
    # 총 리스틔 수는 10개이고 중복 최대값은 5개 모두가 중복되는 10이고 최소값은 0(아무것도 중복되지 않음)   0~10까지의 수로 표현됨
    #print(sum_compare_re)
    
    if sum_compare_re >= 0 and sum_compare_re <= 3:
        in_depth_sent_result = 'weak'
    elif sum_compare_re > 3 and sum_compare_re <= 7:
        in_depth_sent_result = 'somewhat close'
    elif sum_compare_re > 7 :
        in_depth_sent_result = 'very close'
        
        
    # 0.result_emo_list: 문장 + 감성분석결과
    # 1.intendedMoodByPmt : intended mood 
    # 2.detected_mood : 대표 Mood
    # 3.sentence1,sentence2, sentence3 : intended mood vs. your mood 비교결과에 대한 문장생성 커멘트
    # 4.key_emo[:5] : 학생 한명의 에세이에서 추출한 대표감성 5개
    # 5.accepted_essay_av_value : 1000명의 합격한 학생의 대표감성 5개
    # 6.in_depth_sent_result : 최종 심층 분석결과
    # 7.re_mood : 개인 mood 분석 추출결과

    return result_emo_list, intendedMoodByPmt, detected_mood, sentence1, sentence2, sentence3, key_emo[:5], accepted_essay_av_value, in_depth_sent_result, re_mood
                    
                    

def run_first(input_text):
        # 실행하면 : ai_plot_coflict_total_analysis(input_text)  ===>  "Shifts Between Positive and Negative Sentiments" : shifts_btw_neg_pos  ---> 딕셔너리의 값을 어떻게 가져오지?
        ai_plot_coflict_value = ai_plot_coflict_total_analysis(input_text)

        shift_neg_pos_value_re = list(ai_plot_coflict_value.get('Shifts Between Positive and Negative Sentiments'))
        shift_neg_pos_value = shift_neg_pos_value_re[0]
        # print('#############++++++++++++++++++++++++++#########')
        # print('shift_neg_pos_value_re_key : ', shift_neg_pos_value_re[0]) # 키만 가져옴, 이것을 아래 함수에 넣음
        # print('shift_neg_pos_value_re_value : ', shift_neg_pos_value_re[1]) # 값만 가져옴, 이것을 아래 함수에 넣음
        # print('#############++++++++++++++++++++++++++#########')
        # colfict 단어들이 각 문단의 구간에서 얼마나 포함되었는지 계산하여 최대구간, 두번째 많은 구간을 추출할 것
        # conflict 단어 추출
        conflict_words_li_re = list(ai_plot_coflict_value.get("conflict words list"))
        # print('#############++++++++++++++++++++++++++#########')
        # print('conflict_words_li_re: ', conflict_words_li_re) # conflict words 추출한 리스트
        # print('#############++++++++++++++++++++++++++#########')

        return conflict_words_li_re, shift_neg_pos_value


def paragraph_divide_ratio(input_text):

    run_first_result = run_first(input_text)
    conflict_words_li_re = run_first_result[0]
    #print('------------------>conflict_words_li_re:', conflict_words_li_re)

    essay_input_corpus = str(input_text) #문장입력
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
    #print('intro :', intro)
    body_1_ = sentences[intro_n:intro_n + body_1]
    #print('body 1 :', body_1_)
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
    

    ######### conflict 관련 단어 추출 - start #########

    tot_setting_words = conflict_words_li_re

    ######### conflict 관련 단어 추출 - end   #########



    # 구간별 셋팅 단어가 몇개씩 포함되어 있는지 계산 method
    # st_wd : 총 컨플릭 단어들
    # each_parts_ : 구간(intro, body1~3, conclusion)
    def set_wd_conunter_each_parts(st_wd, each_parts_):
        if each_parts_ == intro:
            part_section = 'intro'
        elif each_parts_ == body_1_:
            part_section = 'body #1'
        elif each_parts_ == body_2_:
            part_section = 'body #2'
        elif each_parts_ == body_3_:
            part_section = 'body #3'
        else: #conclusion
            part_section = 'conclusion'
        counter = 0
        for set_itm in st_wd:
            if set_itm in each_parts_:
                counter += 1
            else:
                pass
        return counter, part_section

    # 구간별 셋팅 단어가 몇개씩 포함되어 있는지 계산 
    intro_s_num = set_wd_conunter_each_parts(tot_setting_words, intro)
    #print('intor:', intro_s_num)
    body_1_s_num = set_wd_conunter_each_parts(tot_setting_words, body_1_)
    #print('body1:', body_1_s_num)
    body_2_s_num = set_wd_conunter_each_parts(tot_setting_words, body_2_)
    #print('body2:', body_2_s_num)
    body_3_s_num = set_wd_conunter_each_parts(tot_setting_words, body_3_)
    #rint('body3',body_3_s_num)
    conclusion_s_num = set_wd_conunter_each_parts(tot_setting_words, conclusion)
    #print('conclusion:',conclusion_s_num)

    
    # 가장 많이 포함된 구간을 순서대로 추출
    compare_parts_grup_nums = [] # 숫자와 항복명을 모두 저장(튜플을 리스트로)
    compare_parts_grup_nums_and_parts = [] # 숫자만 리스트로
    
    compare_parts_grup_nums.append(intro_s_num[0])
    compare_parts_grup_nums.append(intro_s_num[1])
    compare_parts_grup_nums_and_parts.append(intro_s_num[0])

    
    compare_parts_grup_nums.append(body_1_s_num[0])
    compare_parts_grup_nums.append(body_1_s_num[1])
    compare_parts_grup_nums_and_parts.append(body_1_s_num[0])
    
    compare_parts_grup_nums.append(body_2_s_num[0])
    compare_parts_grup_nums.append(body_2_s_num[1])
    compare_parts_grup_nums_and_parts.append(body_2_s_num[0])
    
    compare_parts_grup_nums.append(body_3_s_num[0])
    compare_parts_grup_nums.append(body_3_s_num[1])
    compare_parts_grup_nums_and_parts.append(body_3_s_num[0])
    
    compare_parts_grup_nums.append(conclusion_s_num[0])
    compare_parts_grup_nums.append(conclusion_s_num[1])
    compare_parts_grup_nums_and_parts.append(conclusion_s_num[0])
    
    #compare_parts_grup_nums_and_parts =compare_parts_grup_nums_and_parts.sort(reverse=True)
    
    #print('compare_parts_grup: ', compare_parts_grup_nums) # [7, 'intro', 11, 'body #1', 9, 'body #2', 9, 'body #3', 4, 'conclusion']
    
    #순서정렬
    compare_parts_grup_nums_and_parts_sorted = sorted(compare_parts_grup_nums_and_parts, reverse=True)
    #print('compare_parts_grup_nums_and_parts(sorted)', compare_parts_grup_nums_and_parts_sorted) # [11, 9, 9, 7, 4]
    #print('compare_parts_grup_nums_and_parts :',compare_parts_grup_nums_and_parts)
    
    first_result = compare_parts_grup_nums_and_parts_sorted[0]
    second_result = compare_parts_grup_nums_and_parts_sorted[1]
    
    get_first_re = compare_parts_grup_nums.index(first_result) #인덱스 위치찾기
    #print('get_firtst_re:',get_first_re)
    #가장 많은 표현이 들어간 부분 추출(최종값)
    first_snt_part = compare_parts_grup_nums[get_first_re + 1]
    
    get_second_re = compare_parts_grup_nums.index(second_result)
    #print('get_second_re:',get_second_re)
    second_snt_part = compare_parts_grup_nums[get_second_re + 1] # 인덱스 다음 항목이 최종값

    # 결과해석
    # 0.df_sentences: 모든 단어를 데이터프레임으로 변환
    # 1.tot_setting_words: : 추출한 conflict 관련 단어 리스트로 변환
    # 2.first_snt_part: 문단중 가장 conflict 관련 단어가 많은 부분 -> Strength of Tension by Section 문장으로 표현
    # 3.second_snt_part: 문잔중 conflict 관련 단어가 두번째고 많은 부분 -> Strength of Tension by Section 문장으로 표현
    # 4.compare_parts_grup_nums_and_parts : intro body_1 body_2 body_3 conclusion 의 개인 에세이 계산 값
    # 5.conflict_words_li_re[1] : 컨플릭단어추출 분석 결과 리스트

    return df_sentences, tot_setting_words, first_snt_part, second_snt_part, compare_parts_grup_nums_and_parts, conflict_words_li_re


        


def feedback_plot_conflict(prompt_no, ps_input_text):
    ############################################
    #############################################
    # 합격한 학생의 평균 mood 평균 값
    # gruop_input_text_re : 합격한 학생의 평균값
    group_input_text_re = 'Joyful'
    ############################################
    ############################################
    # intended mood
    int_mood_by_prompt_re = intended_mood_by_prompt(prompt_no)
    
    # detected mood
    dtc = ai_emotion_analysis(ps_input_text, prompt_no)
    intended_mood = dtc[1] # intended Mood 
    #print('intended_mood:', intended_mood)
    detected_mood = dtc[2] # detected Mood
    #print('detected_mood:', detected_mood)
    
    # print("추출된 감성 5개: ", dtc[7])
    
    
    # intended value
    if intended_mood == ['disturbed']:
        pc_intended_mood_result = 'disturbed'
        pc_tension = 'Moderate & High Tension'
    elif intended_mood == ['suspenseful']:
        pc_intended_mood_result = 'suspenseful'
        pc_tension = 'High Tension'
    elif intended_mood == ['sad']:
        pc_intended_mood_result = 'sad'
        pc_tension = 'Moderate Tension'
    elif intended_mood == ['joy']:
        pc_intended_mood_result = 'joyful'
        pc_tension = 'Moderate & Low Tension'
    else: # carm
        pc_intended_mood_result = 'carm'
        pc_tension = 'Low Tension'
        
    # Tension Value 계산하기 'Moderate & High Tension', 'High Tension', 'Moderate Tension', 'Moderate & Low Tension', 'Low Tension'
    if detected_mood == ['disturbed']:
        dtc_intended_mood_result = 'disturbed'
        dtc_tension = 'Moderate & High Tension'
    elif detected_mood == ['suspenseful']:
        dtc_intended_mood_result = 'suspenseful'
        dtc_tension = 'High Tension'
    elif detected_mood == ['sad']:
        dtc_intended_mood_result = 'sad'
        dtc_tension = 'Moderate Tension'
    elif detected_mood == ['joy']:
        dtc_intended_mood_result = 'joyful'
        dtc_tension = 'Moderate & Low Tension'
    else: # carm
        dtc_intended_mood_result = 'carm'
        dtc_tension = 'Low Tension'
        
    
    # Intended Mood vs. Plot & Conflict 두개의 값 비교
    if intended_mood == detected_mood:
        comp_int_dtc = '='
    else: # intended_mood vs. detected_mood 같이 않을 경우
        comp_int_dtc = '!=' # 같지 않음

    
    # 문장 생성시작 --> Sentence 1
    if detected_mood == ['disturbed']:
        sentence_1 = ['Your intended mood for the essay is ‘disturbed.’ It means that the story may contain matters that made you feel uneasy or upset. In addition, it may deal with a problem or disagreement and your personal struggle to resolve it. Hence, the plot is likely to be a complex one with multiple emotional fluctuations and conflicts.']
    elif detected_mood == ['suspenseful']:
        sentence_1 = ['Your intended mood for the essay is ‘suspenseful.’ It means that the story may contain multiple elements intertwined with one another. In addition, it may deal with incidents unfolding in a dynamic pattern. Hence, the plot is likely to show a high level of tension with multiple emotional fluctuations and conflicts.']
    elif detected_mood == ['sad']:
        sentence_1 = ['Your intended mood for the essay is ‘sad.’ It means that the story may contain difficult times in life that made you feel grief. In addition, it may deal with your memories that make the readers feel sympathetic. Hence, the plot is likely to be a moderately complex one with some emotional fluctuations and conflicts.']
    elif detected_mood == ['joy']:
        sentence_1 = ['''Your intended mood for the essay is ‘joyful.’ It means that the story may contain elements that made you feel happy. In addition, it may deal with lighthearted life stories and positive lessons you've gained. Hence, the plot is likely to be a pleasant one with moderate emotional fluctuations and conflicts.''']
    else: # calm 
        sentence_1 = ['Your intended mood for the essay is ‘calm.’ It means that the story may contain self-reflection, intellectual topics, and observations that shaped your perspective. Hence, the plot is likely to be somewhat steady with limited emotional fluctuations and conflicts.']
    
    # Sentence 2
    # tension level 계산 ( tension 레벨은 conflict words + sentiment fluctuations (negative to positive to negative 이런거) 를 함께 봅니다. 아마도 conflict words 수 (60%) + sentiment fluctuations (40%) 보면 되지 않을까해요.)
    if dtc_tension == 'High Tension' and intended_mood == ['suspenseful']:
        sentence_2 = ['The detected plot displays a high tension which seems to be', 'closely correlated with', 'your intended mood of the essay.']
    elif dtc_tension == 'High Tension' and intended_mood != ['suspenseful']:
        sentence_2 = ['The detected plot displays a high tension which seems to be', 'somewhat distant from', 'your intended mood of the essay.']
    elif dtc_tension == 'Moderate Tension' and intended_mood == ['disturbed'] or intended_mood == ['sad'] or intended_mood == ['joy']:
        sentence_2 = ['The detected plot displays a moderate tension which seems to be', 'closely correlated with', 'your intended mood of the essay.']
    elif dtc_tension == 'Moderate Tension' and intended_mood != ['disturbed'] and intended_mood != ['sad'] and intended_mood != ['joy']:
        sentence_2 = ['The detected plot displays a moderate tension which seems to be', 'somewhat distant from', 'your intended mood of the essay.']
    elif dtc_tension == 'low tension' and intended_mood == ['carm']:
        sentence_2 = ['The detected plot displays a low tension which seems to be', 'closely correlated with', 'your intended mood of the essay.']
    else:
        sentence_2 = ['The detected plot displays a low tension which seems to be', 'somewhat distant from', 'your intended mood of the essay.']
    
    
    ####  number of stimulus words ####
    # conlifct words 추출, 수량 계산
    plot_conf_re = ai_plot_conf(ps_input_text)
    
    #####################################################
    ################# 합격한 학생의 평균값 ###################
    group_conflict_word_num = 5 # Conflict words numbers
    group_action_verbs_num = 20 # 
    #####################################################
    #####################################################

    count_conflict_list_re = plot_conf_re[5] # conflict words list
    #print('count_conflict_list_re', count_conflict_list_re)
    nums_conflict_words_re = plot_conf_re[6] # conflict words number
    #print('nums_conflict_words_re', nums_conflict_words_re)
    get_words__re = plot_conf_re[7] # Action Verbs list
    #print('get_words__re', get_words__re)
    nums_action_verbs_re = plot_conf_re[8] # Action verbs number
    #print('nums_action_verbs_re', nums_action_verbs_re)
    
    stm_sentence_1 = []
    stm_sentence_2 = []
    
    print("#"*200)
    print("nums_conflict_words_re:",nums_conflict_words_re)
    print("group_conflict_word_num:",group_conflict_word_num)
    print("nums_action_verbs_re:",nums_action_verbs_re)
    print("group_action_verbs_num:",group_action_verbs_num)
    print("#"*200)
    
    #######  nums_conflict_words_re: 0   #######
    #######  group_conflict_word_num: 5   #######
    #######  nums_action_verbs_re: 36   #######
    #######  group_action_verbs_num: 20   #######
    

    # 개인, 그룹의 결과 비교,  컨플릭단어와 액션동사 사용량 비교
    if nums_conflict_words_re == group_conflict_word_num and nums_action_verbs_re == group_action_verbs_num:
        stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', 'similar number of', 'conflict oriented words and', 'similar number of', 'action verbs in your story.']
        stm_sentence_2 = ['''Overall, your plot's tension level looks good compared with the accepted average.''']
    elif nums_conflict_words_re > (group_conflict_word_num - group_conflict_word_num * 0.3)  and nums_conflict_words_re < (group_conflict_word_num + group_conflict_word_num * 0.3): 
                                                                                                                
        if nums_action_verbs_re > (group_action_verbs_num - group_action_verbs_num * 0.3) and nums_action_verbs_re < (group_action_verbs_num + group_action_verbs_num * 0.3):
            stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', 'similar number of', 'conflict oriented words and', 'similar number of', 'action verbs in your story.']
            stm_sentence_2 = ['''Overall, your plot's tension level looks good compared with the accepted average.''']
        else:
            pass
                                                                                                                    
    elif nums_conflict_words_re < (group_conflict_word_num + group_conflict_word_num * 0.3)  and nums_action_verbs_re < (group_action_verbs_num + group_action_verbs_num * 0.3):
        
        if nums_action_verbs_re < (group_action_verbs_num + group_action_verbs_num * 0.3) and nums_action_verbs_re > (group_action_verbs_num - group_action_verbs_num * 0.3):                                                                                                           
            stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', 'similar number of', 'conflict oriented words and', 'similar number of', 'action verbs in your story.']
            stm_sentence_2 = ['''Overall, your plot's tension level looks good compared with the accepted average.''']
        else:
            pass

    elif nums_conflict_words_re > group_conflict_word_num and nums_action_verbs_re > group_action_verbs_num:
        first_value = abs(nums_conflict_words_re - group_conflict_word_num)
        second_value = abs(nums_action_verbs_re - group_action_verbs_num)
        stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', first_value, 'more', 'conflict oriented words and', second_value,'more', 'action verbs in your story.']
        stm_sentence_2 = ['Overall, you may consider', 'using less' , 'words to', 'alleviate', '''the plot's tension''']

    elif nums_conflict_words_re > group_conflict_word_num and nums_action_verbs_re < group_action_verbs_num:
        first_value = abs(nums_conflict_words_re - group_conflict_word_num)
        second_value = abs(nums_action_verbs_re - group_action_verbs_num)
        stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', first_value, 'more', 'conflict oriented words and', second_value,'fewer', 'action verbs in your story.']
        stm_sentence_2 = ['Overall, you may consider', 'using less' , 'words to', 'intensify', '''the plot's tension''']

    elif nums_conflict_words_re < group_conflict_word_num and nums_action_verbs_re > group_action_verbs_num:
        first_value = abs(nums_conflict_words_re - group_conflict_word_num)
        second_value = abs(nums_action_verbs_re - group_action_verbs_num)
        stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', first_value, 'fewer', 'conflict oriented words and', second_value,'more', 'action verbs in your story.']
        stm_sentence_2 = ['Overall, you may consider', 'adding more' , 'words to', 'alleviate', '''the plot's tension''']

    elif nums_conflict_words_re < group_conflict_word_num and nums_action_verbs_re < group_action_verbs_num:
        first_value = abs(nums_conflict_words_re - group_conflict_word_num)
        second_value = abs(nums_action_verbs_re - group_action_verbs_num)
        stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', first_value, 'fewer', 'conflict oriented words and', second_value,'fewer', 'action verbs in your story.']
        stm_sentence_2 = ['Overall, you may consider', 'adding more' , 'words to', 'intensify', '''the plot's tension''']
    else:  ##### comment가 없는 경우 체크 kjp
        pass

    def ferq_moder_spar(value_no):
        re_shift_cnt = abs(value_no)#절대값
        if re_shift_cnt > 10: #감정기복이 어느 한쪽으로 편향이 6번이상 된다면 >>>>>>>>>>>>>>> !! 기준값 결정하는것 중요(group 평균값 계산해서 적용할 것)
            shift_emo_ratio = 'sparse'
        elif re_shift_cnt <= 9 and re_shift_cnt > 5:
            shift_emo_ratio = 'moderate'
        else: #5이하라면 감정기복이 심함 pos neg가 많이교차됨(반반일 경우)
            shift_emo_ratio = 'frequent'

        return shift_emo_ratio

    # Shifts Between Positive and Negative Sentiments 문장 생성 부분 (def shifts_Bt_PoNe(input_val))

    # 실행하면 : ai_plot_coflict_total_analysis(input_text))   ===>  "Shifts Between Positive and Negative Sentiments" : shifts_btw_neg_pos  
    
    # Shifts Between Positive and Negative Sentiments 문장 생성 부분
    # print("ps_input_text:",ps_input_text)
    result_=paragraph_divide_ratio(ps_input_text)
    shift_neg_pos_value = result_[5]

    def shifts_Bt_PoNe(prompt_no, shift_neg_pos_value): #shift_neg_pos_value는 하나의 키값(단어) 입력됨
        ###############################################################################
        ###############################################################################
        # prompt 별로 평균합격한 학생들의 감정기복비율 (임의로 정함, 나중에 계산해서 적용할 것)
        if prompt_no == 'ques_one':
            group_pmt_no_value = 2 # 이 숫자들을 다시 frequent 여부를 계산해야 함, 그럴려면 위 함수를 재활용해야 함
            re_fms = ferq_moder_spar(group_pmt_no_value) # 문항의 결과값 frequent / moderate / sparse 중 택1
            shift_between_pos_neg_sentiments_admitted = "Moderate"
        elif prompt_no == 'ques_two':
            group_pmt_no_value = 2
            re_fms = ferq_moder_spar(group_pmt_no_value) # 문항의 결과값 frequent / moderate / sparse 중 택1
            shift_between_pos_neg_sentiments_admitted = "Frequent"
        elif prompt_no == 'ques_three':
            group_pmt_no_value = 3
            re_fms = ferq_moder_spar(group_pmt_no_value) # 문항의 결과값 frequent / moderate / sparse 중 택1
            shift_between_pos_neg_sentiments_admitted = "Frequent"
        elif prompt_no =='ques_four':
            group_pmt_no_value = 4
            re_fms = ferq_moder_spar(group_pmt_no_value) # 문항의 결과값 frequent / moderate / sparse 중 택1
            shift_between_pos_neg_sentiments_admitted = "Moderate"
        elif prompt_no == 'ques_five':
            group_pmt_no_value = 5
            re_fms = ferq_moder_spar(group_pmt_no_value) # 문항의 결과값 frequent / moderate / sparse 중 택1
            shift_between_pos_neg_sentiments_admitted = "Moderate"
        elif prompt_no == 'ques_six':
            group_pmt_no_value = 3
            re_fms = ferq_moder_spar(group_pmt_no_value) # 문항의 결과값 frequent / moderate / sparse 중 택1
            shift_between_pos_neg_sentiments_admitted = "Sparse"
        else: # prompt_no = 'prompt_7':
            group_pmt_no_value = 0
            re_fms = ferq_moder_spar(group_pmt_no_value) # 문항의 결과값 frequent / moderate / sparse 중 택1
            shift_between_pos_neg_sentiments_admitted = "Sparse"
        ###############################################################################
        if re_fms == shift_neg_pos_value:
            sentence_2_val = 'match'
        else: # re_fms != shift_neg_pos_value
            sentence_2_val = 'does not match'

        return shift_neg_pos_value, sentence_2_val,re_fms,shift_between_pos_neg_sentiments_admitted

    # 문장생성
    Sft_BT_PoNe_re = shifts_Bt_PoNe(prompt_no, shift_neg_pos_value)
    
    sentence_max = dict(Counter(Sft_BT_PoNe_re[0]))
    
    
    
    
    sentense_1_PN = ['The admitted cases tend to display', list(sentence_max.keys())[:2],'fluctuations between polarizing sentiments throughout the story for this type of essay prompt.']
    sentense_2_PN = ['Such pattern observed from the admitted case average seems to', Sft_BT_PoNe_re[1], 'with the pattern detected from your personal statement.']

    # 문장생성
    # Strength of Tension by Section
    # StrengthTensionBySection
    
    indicator_sentiment_shifits_pos_neg = ""

    sentence_1_STS = ['Both physical and emotional conflicts and fluctuations in the plot constitute the ‘ups-and-downs’ which add excitement to the story.']
    
    result_conf_part_1st_2nd = paragraph_divide_ratio(ps_input_text) #[2], [3]이 순서대로 많은 구간임
   
    first = result_conf_part_1st_2nd[2] # conflict 값이 가장 많은 구간
    second = result_conf_part_1st_2nd[3] # conflict 값이 두번째로 많은 구간
    each_parts_of_conflict_words_used = result_conf_part_1st_2nd[4] # 구간별 컨플릭 단어 리스트 intro body1~3 conclusion

    sentence_2_STS = ['Dividing up the personal statement in 5 equal parts by the word count, AI analysis indicated that highest levels of tension are concentrated in', first, 'and', second, 'of the accepted case average.']


    # 문장생성
    # 각 구간의 셋팅 관련 표현의 합격자 평균값(임의로 넣음, 나중에 평균값을 계산해서 적용해야 함)
    ##########################################################
    ##########################################################
    group_conflict_words_parts_mean_value = [] # intro, body 1~3, conclusion
    ##########################################################
       # prompt별로 셋팅 단어 적용 비율 
    pmt_1 = [3, 2, 2, 2, 1]
    pmt_2 = [6, 4, 2, 2, 1]
    pmt_3 = [5, 3, 2, 2, 2]
    pmt_4 = [2, 2, 2, 1, 2]
    pmt_5 = [3, 2, 1, 1, 2]
    pmt_6 = [2, 1, 1, 1, 1]
    ##########################################################
    if prompt_no == 'ques_one':
        group_conflict_words_parts_mean_value = pmt_1
    
    elif prompt_no == 'ques_two':
        group_conflict_words_parts_mean_value = pmt_2

    elif prompt_no == 'ques_three':
        group_conflict_words_parts_mean_value = pmt_3

    elif prompt_no == 'ques_four':
        group_conflict_words_parts_mean_value = pmt_4

    elif prompt_no == 'ques_five':
        group_conflict_words_parts_mean_value = pmt_5

    else : #prompt_no == 'ques_one':
        group_conflict_words_parts_mean_value = pmt_6


    # each_parts_of_conflict_words_used # 개인의 구간별 컨플릭 단어 사용 수 리스트!

    # 각각의 값을 비교하고, 0.3 의 오차범위에서 같으면 True 
    def compart(val_1, val_2):
        if val_1 < (val_2 + val_2 * 0.3) and val_1 > (val_2 - val_2 * 0.3):
            result_compart = True
        else:
            result_compart = False
        return result_compart

    # 구간별 두개의 값(그룹, 개인) 비교 함수
    def comp_each_parts(personal, group):
        if personal == group: # 각 구간의 값이 일치하면
            over_sentence_4 = ['Comparing this with your essay, we see a very similar pattern.']  
        elif compart(personal[0], group[0]) and compart(personal[1], group[1]) and compart(personal[2], group[2]) and compart(personal[3], group[3]): # 30% 범위 내로 각 값이 같다면       
            over_sentence_4 = ['Comparing this with your essay, we see a very similar pattern.'] 
            
        elif personal[1] + personal[2] == group[1] + group[2]: # body1 + body 2 로 개인과 그릅울 비교
            over_sentence_4 = ['Comparing this with your essay, we see some similarities in the pattern.']
        elif personal[1] + personal[2] < (group[1] + group[2]) + (group[1] + group[2]) * 0.3:
            over_sentence_4 = ['Comparing this with your essay, we see some similarities in the pattern.']
        elif personal[1] + personal[2] > (group[1] + group[2]) - (group[1] + group[2]) * 0.3:
            over_sentence_4 = ['Comparing this with your essay, we see some similarities in the pattern.']
        else: # 각 구간들이 불일치
            over_sentence_4 = ['Comparing this with your essay, we see a different pattern.']
        return over_sentence_4       
    # 최종 문장생성
    sentence_3_STS = comp_each_parts(each_parts_of_conflict_words_used, group_conflict_words_parts_mean_value)

    ###############
    ###  결과해석 ###
    ###############

    ## option  - 추가분석 시 아래 두개의 분석결과 사용해도됨, 현재코드에서는 불필요함 ##
    # pc_tension : 개인의 선택한 tension 결과
    # intended_mood : 개인의 에세이를 prompt 문항에 의해 분석한 결과 

   

    result_data = {
        "intended_mood" : pc_intended_mood_result, #intended mood by you  --> 개인이 선택한 intended mood 선택 결과
        "detected_mood" : detected_mood[0], #Detected Plot Complexity & Conflicts --> 개인의 에세이를 감성분석한 결과
       
        "stimulus_conflict_words_admmitted" : group_conflict_word_num,#합격한 학생들의 컨플릭 단어 사용량
        "stimulus_conflict_words_your_essay" : nums_conflict_words_re, #개인의 컨플릭 단어 사용량
        "stimulus_action_verbs_admmitted" : group_action_verbs_num, #합격한 학생들의 Action Verbs 사용량
        "stimulus_action_verbs_your_essay" : nums_action_verbs_re, #개인의 Action Verbs 사용량
        
        "intended_mood_plot_conflict_comment_1" : sentence_1, #문장생성
        "intended_mood_plot_conflict_comment_2" : sentence_2, #문장생성
        
        "stimulus_words_comment_1" : stm_sentence_1,#문장생성
        "stimulus_words_comment_2" : stm_sentence_2,#문장생성
        
        "shift_between_pos_neg_comment_1" : sentense_1_PN,#문장생성
        "shift_between_pos_neg_comment_2" : sentense_2_PN,#문장생성
        
        "strength_tension_comment_1" : sentence_1_STS,#문장생성
        "strength_tension_comment_2" : sentence_2_STS,#문장생성
        "strength_tension_comment_3" : sentence_3_STS,#문장생성
        
        # strength_tension_admitted_case_list = strength_tension_admitted_case_list,  ### kjp 이부분 값을 부탁드립니다. 
        # strength_tension_your_essay_list = strength_tension_your_essay_list,        ### kjp 이부분 값을 부탁드립니다. 
        "shift_between_pos_neg_sentiments_admitted" : Sft_BT_PoNe_re[3],
        
        "shift_emo_ratio" : Sft_BT_PoNe_re[2],  #### 
        
        "indicator_conflict_word" :  dict(Counter(count_conflict_list_re)), # conflict words list ------> 웹페이지에 표시해야 함
        "indicator_action_verbs" : dict(Counter(get_words__re)), #Action Verbs list -------> 웹페이지에 표시해야 함
        
        
        
        "indicator_sentiment_shifits_pos_neg" : dict(Counter(plot_conf_re[10]))
    }
        
    return result_data


## 실행 ###
# - 입력값 - #

# prompt_no = 'ques_one' #이런 형식으로 넣어야 함

# input_text = """My hand lingered on the cold metal doorknob. I closed my eyes as the Vancouver breeze ran its chilling fingers through my hair. The man I was about to meet was infamous for demanding perfection. But the beguiling music that faintly fluttered past the unlatched window’s curtain drew me forward, inviting me to cross the threshold. Stepping into the apartment, under the watchful gaze of an emerald-eyed cat portrait, I entered the sweeping B Major scale.

# Led by my intrinsic attraction towards music, coupled with the textured layers erupting the instant my fingers grazed the ivory keys, driving the hammers to shoot vibrations up in the air all around me, I soon fell in love with this new extension of my body and mind. My mom began to notice my aptitude for piano when I began returning home with trophies in my arms. These precious experiences fueled my conviction as a rising musician, but despite my confidence, I felt like something was missing.

# Back in the drafty apartment, I smiled nervously and walked towards the piano from which the music emanated. Ian Parker, my new piano teacher, eyes-closed and dressed in black glided his hands effortlessly across the keys. I stood beside a leather chair, waiting as he finished the phrase. He stood up. I sat down.

# Chopin Black Key Etude — a piece I knew so well I could play it eyes-closed. I took a breath and positioned my right hand in a G-flat 2nd inversion. 
# Just one measure in, I was stopped. 
# 	“Start again.”
# 	Taken by surprise, I spun left. His eyes were on the score, not me. 
# 	I started again. Past the first measure, first phrase, then stopped again. What is going on? 
	
# 	“Are you listening?”
# I nodded. Of course I am. 
# “But are you really listening?”

# As we slowly dissected each measure, I felt my confidence slip away. The piece was being chipped into fragments. Unlike my previous teachers, who listened to a full performance before giving critical feedback, Ian stopped me every five seconds. One hour later, we only got through half a page. 

# Each consecutive week, the same thing happened. I struggled to meet his expectations. 
# “I’m not here to teach you just how to play. I’m here to teach you how to listen.” 
# I realized what Ian meant — listening involves taking what we hear and asking: is this the sound I want? What story am I telling through my interpretation? 

# Absorbed in the music, I allowed my instincts and muscle memory to take over, flying past the broken tritones or neapolitan chords. But even if I was playing the right notes, it didn’t matter. Becoming immersed in the cascading arpeggio waterfalls, thundering basses, and fairydust trills was actually the easy part, which brought me joy and fueled my love for music in the first place. However, music is not just about me. True artists perform for their audience, and to bring them the same joy, to turn playing into magic-making, they must listen as the audience. 

# The lesson Ian taught me echoes beyond practice rooms and concert halls. I’ve learned to listen as I explore the hidden dialogue between voices, to pauses and silence, equally as powerful as words. Listening is performing as a soloist backed up by an orchestra. Listening is calmly responding during heated debates and being the last to speak in a SPS Harkness discussion. It’s even bouncing jokes around the dining table with family. I’ve grown to envision how my voice will impact the stories of those listening to me.

# To this day, my lessons with Ian continue to be tough, consisting of 80% discussion and 20% playing. When we were both so immersed in the music that I managed to get to the end of the piece before he looked up to say, “Bravo.” Now, even when I practice piano alone, I repeat my refrain: Are you listening?  """

# input_text = "Reflect on a time when you challenged a belief or idea. What prompted you to act? Would you make the same decision again? My shaking fingers closed around the shiny gold pieces of the saxophone in its case, leaving a streak of fingerprints down the newly cleaned exterior. The soft and melodic sound of a classic jazz ballad floated out of a set of speakers to my right, mixing with the confident chatter of students in the back row. As usual, I shuffled around in the back of the classroom, attempting to blend in with the sets of cubbies. With my confidence already fading, I sat and thought to myself, I wonder if it’s too late to drop this course? I heard Art Exploration still has plenty of openings! The director stepped to the front of the room, and snapped his fingers in a slow rhythm that the drummer tapped out with his worn wooden sticks. The band congregated in the middle of the room, each person tapping his or her feet along with the drummer. Just relax! I scolded myself, just be jazzy and no one will notice you look out of place. I attempted to mimic the laid-back, grooving movements of my peers, but to no avail. My movements were about as far away from ‘jazzy’ as one could possibly get. I was used to the rigid accents and staccatos of the concert band world. As the solo section began to move around the room most students seemed relaxed and loose, acting as if they were easygoing musicians on a street corner in New Orleans. I felt my body tense and my eyes dart nervously around the room for a quick escape route. My confidence plummeted as the person in front of me swung a closing riff and looked expectantly in my direction. I quickly pressed the mouthpiece to my tongue and played a few stiff sounding riffs, but the notes seemed to fall flat. I felt my face flush a nice shade of deep red as I stood in the middle of the room with the drummer tapping away, waiting for me to step in and finish. All of a sudden, the director shouted, ‘Play a riff! Dance! Come on, Phoebe, at least do something!’ His exasperated tone prompted me to attempt to salvage the bleak looking situation I was currently facing."

# data = feedback_plot_conflict(prompt_no,input_text)

# key_value_print(data)


####################################################################################################

# intended_mood :  joyful
# detected_mood :  suspenseful
# stimulus_conflict_words_admmitted :  5
# stimulus_conflict_words_your_essay :  76
# stimulus_action_verbs_admmitted :  20
# stimulus_action_verbs_your_essay :  19
# intended_mood_plot_conflict_comment_1 :  ['Your intended mood for the essay is ‘suspenseful.’ It means that the story may contain multiple elements intertwined with one another. In addition, it may deal with incidents unfolding in a dynamic pattern. Hence, the plot is likely to show a high level of tension with multiple emotional fluctuations and conflicts.']
# intended_mood_plot_conflict_comment_2 :  ['The detected plot displays a high tension which seems to be', 'somewhat distant from', 'your intended mood of the essay.']
# stimulus_words_comment_1 :  ['Compared to the accepted case average for this prompt, you have spent', 71, 'more', 'conflict oriented words and', 1, 'fewer', 'action verbs in your story.']
# stimulus_words_comment_2 :  ['Overall, you may consider', 'using less', 'words to', 'intensify', "the plot's tension"]
# shift_between_pos_neg_comment_1 :  ['The admitted cases tend to display', ['enmity', 'friction'], 'fluctuations between polarizing sentiments throughout the story for this type of essay prompt.']
# shift_between_pos_neg_comment_2 :  ['Such pattern observed from the admitted case average seems to', 'does not match', 'with the pattern detected from your personal statement.']
# strength_tension_comment_1 :  ['Both physical and emotional conflicts and fluctuations in the plot constitute the ‘ups-and-downs’ which add excitement to the story.']
# strength_tension_comment_2 :  ['Dividing up the personal statement in 5 equal parts by the word count, AI analysis indicated that highest levels of tension are concentrated in', 'intro', 'and', 'intro', 'of the accepted case average.']
# strength_tension_comment_3 :  ['Comparing this with your essay, we see some similarities in the pattern.']
# shift_emo_ratio :  frequent
# indicator_conflict_word :  {'enmity': 1, 'friction': 1, 'divergent': 1, 'varying': 1, 'incongruous': 1, 'bone of contention': 1, 'variance': 1, 'sore point': 1, 'issue': 1, 'beef': 1, 'differ': 1, 'discord': 1, 'combat': 1, 'question': 1, 'incongruity': 1, 'fight': 1, 'at variance': 1, 'contrariety': 1, 'disagreeing': 1, 'striving': 1, 'discrepant': 1, 'contention': 1, 'bone to pick': 1, 'rivalry': 1, 'battle': 1, 'dispute': 1, 'incompatible': 1, 'inconsistent': 1, 'collision': 1, 'dissension': 1, 'contrary': 1, 'agitation': 1, 'controversy': 1, 'conflicting': 1, 'set-to': 1, 'contrasting': 1, 'point in question': 1, 'debate': 1, 'hatred': 1, 'war': 1, 'in opposition': 1, 'diverge': 1, 'opposite': 1, 'vary': 1, 'conflict': 1, 'contradictory': 1, 'irreconcilable': 1, 'resistance': 1, 'at odds': 1, 'different': 1, 'competition': 1, 'discordant': 1, 'conflicted': 1, 'disagree': 1, 'differing': 1, 'tug-of-war': 1, 'contrast': 1, 'strife': 1, 'conflicts': 1, 'clash': 1, 'rancor': 1, 'clashing': 1, 'matter': 1, 'quarrel': 1, 'disagreement': 1, 'opposing': 1, 'engagement': 1, 'opposition': 1, 'contest': 1, 'fray': 1, 'collide': 1, 'emulation': 1, 'matter at hand': 1, 'hostility': 1, 'antithetical': 1, 'fracas': 1}
# indicator_action_verbs :  {'act': 1, 'make': 1, 'set': 1, 'blend': 1, 'drop': 1, 'relax': 1, 'place': 1, 'mimic': 1, 'get': 1, 'move': 1, 'dart': 1, 'escape': 1, 'fall': 1, 'flush': 1, 'step': 1, 'finish': 1, 'play': 1, 'dance': 1, 'do': 1}
# indicator_sentiment_shifits_pos_neg :  {' The soft and melodic sound of a classic jazz ballad floated out of a set of speakers to my right, mixing with the confident chatter of students in the back row': 1, ' I attempted to mimic the laid-back, grooving movements of my peers, but to no avail': 1, ' My movements were about as far away from ‘jazzy’ as one could possibly get': 1, ' I was used to the rigid accents and staccatos of the concert band world': 1, ' As the solo section began to move around the room most students seemed relaxed and loose, acting as if they were easygoing musicians on a street corner in New Orleans': 1, ' I felt my body tense and my eyes dart nervously around the room for a quick escape route': 1, ' My confidence plummeted as the person in front of me swung a closing riff and looked expectantly in my direction': 1, ' All of a sudden, the director shouted, ‘Play a riff! Dance! Come on, Phoebe, at least do something!’ His exasperated tone prompted me to attempt to salvage the bleak looking situation I was currently facing': 1}

# ####################################################################################################



# print("pc_intended_mood_result(intended mood by you  --> 개인이 선택한 intended mood 선택 결과):" , result["pc_intended_mood_result"]) #intended mood by you  --> 개인이 선택한 intended mood 선택 결과
# print("detected_mood(Detected Plot Complexity & Conflicts --> 개인의 에세이를 감성분석한 결과):" , result["detected_mood"]) #Detected Plot Complexity & Conflicts --> 개인의 에세이를 감성분석한 결과
# print("comp_int_dtc(intended Mood vs. Plot & Conflict 부분의 두 값 비교로 --> 같거(=)나 같지 않음(!=)):" , result["comp_int_dtc"]) #intended Mood vs. Plot & Conflict 부분의 두 값 비교로 --> 같거(=)나 같지 않음(!=)
# print()
# print("group_conflict_word_num(합격한 학생들의 컨플릭 단어 사용량):" , result["group_conflict_word_num"])#합격한 학생들의 컨플릭 단어 사용량
# print("nums_conflict_words_re(개인의 컨플릭 단어 사용량):" , result["nums_conflict_words_re"]) #개인의 컨플릭 단어 사용량
# print("group_action_verbs_num(합격한 학생들의 Action Verbs 사용량):" , result["group_action_verbs_num"]) #합격한 학생들의 Action Verbs 사용량
# print("nums_action_verbs_re(개인의 Action Verbs 사용량):" , result["nums_action_verbs_re"]) #개인의 Action Verbs 사용량
# print()
# print("sentence_1:" , result["sentence_1"]) #문장생성
# print("sentence_2:" , result["sentence_2"]) #문장생성
# print("stm_sentence_1:" , result["stm_sentence_1"])#문장생성
# print("stm_sentence_2:" , result["stm_sentence_2"])#문장생성
# print("sentense_1_PN:" , result["sentense_1_PN"])#문장생성
# print("sentense_2_PN:" , result["sentense_2_PN"])#문장생성
# print("sentence_1_STS:" , result["sentence_1_STS"])#문장생성
# print("sentence_2_STS:" , result["sentence_2_STS"])#문장생성
# print("sentence_3_STS:" , result["sentence_3_STS"])#문장생성
# print()
# print("count_conflict_list_re(conflict words list ------> 웹페이지에 표시해야 함):" ,  result["count_conflict_list_re"]) # conflict words list ------> 웹페이지에 표시해야 함
# print("get_words__re(Action Verbs list -------> 웹페이지에 표시해야 함):" , result["get_words__re"]) #Action Verbs list -------> 웹페이지에 표시해야 함
# print("#"*100)