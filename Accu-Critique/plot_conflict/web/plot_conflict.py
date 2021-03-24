
#conflict
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
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
nltk.download('stopwords')
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
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline

tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
)


input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""



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
        filtered_chr_text = []
        for k in token_input_text:
            for j in confict_words_list:
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

#         import pandas as pd

#         df_conf_words = pd.DataFrame(ext_sim_words_key, columns=['words','values']) #데이터프레임으로 변환
#         df_r = df_conf_words['words'] #words 컬럼 값 추출
#         ext_sim_words_key = df_r.values.tolist() # 유사단어 추출
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
    print('token_list_str:', token_list_str)
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
    print('confict_words_list:', confict_words_list_set)
    
    # 문장에 들어있는 추출된 conflict 단어들 : count_conflict_list ==================> conflict 단어가 없음(겹치는 단어 없나?)
    count_conflict_list = []
    for ittm in confict_words_list_set:
        if ittm in token_list_str_set:
            count_conflict_list.append(ittm)
            
    print('문장에 들어있는 추출된 conflict 단어들:', count_conflict_list)
    
    # 전체문장에 들어있는 conflict 단어 수
    nums_conflict_words =  len(count_conflict_list)
    print('전체문장에 들어있는 conflict 단어 수:', nums_conflict_words)
    
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
    data_action_verbs = pd.read_csv('actionverbs.csv')
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
    print('Action Verbs:', get_words__)
    nums_action_verbs = len(get_words__)
    print('Number of Action Verbs:', nums_action_verbs)


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

    print ("ACTION VERBS RATIO :", action_verbs_ratio )


    #########################################################################
    # 11. 글속에 감정이 얼마나 표현되어 있는지 분석 - origin (Bert pre trained model 활용)
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


    #중립적인 감정을 제외하고, 입력한 문장에서 다양한 감정을 모두 추출하고 어떤 감정이 있는지 계산해보자
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
        re_shift_cnt = abs(shift_counter)#절대값
        if re_shift_cnt > 10: #감정기복이 어느 한쪽으로 편향이 6번이상 된다면 >>>>>>>>>>>>>>> !! 기준값 결정하는것 중요(group 평균값 계산해서 적용할 것)
            shift_emo_ratio = 'sparse'
        elif re_shift_cnt <= 9 and re_shift_cnt > 5:
            shift_emo_ratio = 'moderate'
        else: #5이하라면 감정기복이 심함 pos neg가 많이교차됨(반반일 경우)
            shift_emo_ratio = 'frequent'
        
        return shift_emo_ratio
    
    
    
    
    ## 감정분석한 원본 문장 추출하여 웹페이지에 출력(editer 기능 구현 부분)
    

    #중립감정 제거 및 유일한 감정값 확인
    #unique
    unique_re = set(unique) #중복제거

    ############################################################################
    # 글에 표현된 감정이 얼마나 다양한지 분석 결과!!!¶
    print("====================================================================")
    print("에세이에 표현된 다양한 감정 수:", len(unique_re))
    print("====================================================================")

    #분석가능한 감정 총 감정 수 - Bert origin model 적용시 28개 감정 추출돰
    total_num_emotion_analyzed = 28

    # 감정기복 비율 계산 !!!
    result_emo_swings =round(len(unique_re)/total_num_emotion_analyzed *100,1) #소숫점 첫째자리만 표현
    result_emo_swings
    print("문장에 표현된 감정 비율 : ", result_emo_swings)
    print("====================================================================")


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
                            'darkness', 'shadow', 'depths', 'void','maze', 'labyrinth'
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
        print (filtered_setting_text__) # 중복값 제거 확인
        
#         for i in filtered_setting_text__:
#             ext_setting_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
        setting_total_count = len(filtered_setting_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현 수
        setting_count_ = len(filtered_setting_text__) #중복제거된 setting표현 총 수
            
        result_setting_words_ratio = round(setting_total_count/total_words * 100, 2)
        #return result_setting_words_ratio, total_sentences, total_words, setting_total_count, setting_count_, ext_setting_sim_words_key
        return result_setting_words_ratio


    # 셋팅 비율 계산
    settig_ratio_re = setting_anaysis(input_text)
    print("====================================================================")
    print("SETTING RATIO : ", settig_ratio_re)
    print("====================================================================")


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
    print("전체 문장에서 캐릭터를 의미하는 단어나 유사어 비율 :", character_ratio_result)

    ###########################################################
    ############# Degree of Conflict  비율 계산 #################
    conflict_word_ratio = round(len(count_conflict_list) / len(input_text_list) * 1000, 1)  
    print("Degree of conflict  단어가 전체 문장(단어)에서 차지하는 비율 계산 :", conflict_word_ratio)

    global coflict_ratio
    coflict_ratio = [conflict_word_ratio] #그래프로 표현하는 값



    ###########################################################
    ############# Emotional Rollercoaster  비율 계산 #################
    print("감정기복비율 :", result_emo_swings) 

    # 셋팅비율 계산
    print("셋팅비율 계산 : ", settig_ratio_re)

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
    print("Plot Complxity :", st_input )

    global plot_comp_ratio

    plot_comp_ratio = [round(st_input, 2)]
    
    
    ## 감정기복 분석 --> plot & conflict 의 문장 생성부분
    ext_shift_emoiton_re = ext_shift_emoiton_ratio(unique)

    print("===============================================================================")
    print("======================      Degree of Conflict   ==============================")
    print("===============================================================================")

    
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
    
    return st_input, result_emo_swings, conflict_word_ratio, df_sent, graph_calculation_list, count_conflict_list, nums_conflict_words, get_words__, nums_action_verbs, ext_shift_emoiton_re




def ai_plot_coflict_total_analysis(input_text):

    plot_conf_re = ai_plot_conf(input_text)
    
    #count_conflict_list_re = plot_conf_re[5] # conflict words list
    #nums_conflict_words_re = plot_conf_re[6] # conflict words number
    #get_words__re = plot_conf_re[7] # Action Verbs list
    #nums_action_verbs_re = plot_conf_re[8] # Action verbs number
    
    #Shifts Between Positive and Negative Sentiments의 감정기복 값 계산 결과로 문장생성 frequent / moderate / sparse
    plot_conf_re[9]
    
    print("1명의 에세이 결과 계산점수 :", plot_conf_re)
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
        print('min_', min_)
        max_ = int(ideal_mean+ideal_mean*0.6)
        print('max_: ', max_)
        div_ = int(((ideal_mean+ideal_mean*0.6)-(ideal_mean-ideal_mean*0.6))/3)
        print('div_:', div_)

        #결과 판단 Lacking, Ideal, Overboard
        cal_abs = abs(ideal_mean - one_ps_char_desc) # 개인 - 단체 값의 절대값계산

        print('cal_abs 절대값 :', cal_abs)
        compare7 = (one_ps_char_desc + ideal_mean)/6
        compare6 = (one_ps_char_desc + ideal_mean)/5
        compare5 = (one_ps_char_desc + ideal_mean)/4
        compare4 = (one_ps_char_desc + ideal_mean)/3
        compare3 = (one_ps_char_desc + ideal_mean)/2
        print('compare7 :', compare7)
        print('compare6 :', compare6)
        print('compare5 :', compare5)
        print('compare4 :', compare4)
        print('compare3 :', compare3)



        if one_ps_char_desc > ideal_mean: # 개인점수가 평균보다 클 경우는 overboard
            if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
                print("Overboard: 2")
                result = 2 #overboard
                score = 1
            elif cal_abs > compare4: # 28
                print("Overvoard: 2")
                result = 2
                score = 2
            elif cal_abs > compare5: # 22
                print("Overvoard: 2")
                result = 2
                score = 3
            elif cal_abs > compare6: # 18
                print("Overvoard: 2")
                result = 2
                score = 4
            else:
                print("Ideal: 1")
                result = 1
                score = 5
        elif one_ps_char_desc < ideal_mean: # 개인점수가 평균보다 작을 경우 lacking
            if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
                print("Lacking: 2")
                result = 0
                score = 1
            elif cal_abs > compare4: # 28
                print("Lacking: 2")
                result = 0
                score = 2
            elif cal_abs > compare5: # 22
                print("Lacking: 2")
                result = 0
                score = 3
            elif cal_abs > compare6: # 18
                print("Lacking: 2")
                result = 0
                score = 4
            else:
                print("Ideal: 1")
                result = 1
                score = 5
                
        else:
            print("Ideal: 1")
            result = 1
            score = 5

        return result, score


    plot_complexity_result = lackigIdealOverboard(plot_complexity_mean, plot_complexity)
    emotional_rollercoaster_result = lackigIdealOverboard(emotional_rollercoaster_mean, emotional_rollercoaster)
    degree_conflict_result = lackigIdealOverboard(degree_conflict_mean, degree_conflict)

    fin_result = [plot_complexity_result, emotional_rollercoaster_result, degree_conflict_result]
    print("fin_result:", fin_result)  # [(0:lacking, 1:score), (0:lacking, 2:score), (2:overboard, 1:score)]

    each_fin_result = [fin_result[0][0], fin_result[1][0], fin_result[2][0]]

    # 최종 character  전체 점수 계산
    overall_character_rating = [round((fin_result[0][1]+ fin_result[1][1] + fin_result[2][1])/3,2)]

    result_final = each_fin_result + overall_character_rating + group_db_fin_result_plot + coflict_ratio + plot_comp_ratio

    df_sent = plot_conf_re[3]
    
    neg =  list(map(float, df_sent["neg"]))
    neu =  list(map(float, df_sent["neu"]))
    pos =  list(map(float, df_sent["pos"]))
    compound =  list(map(float, df_sent["compound"]))
    
    print(df_sent)
    
    print("neg>>>>>",neg)
    print("neu>>>>>",neu)
    print("pos>>>>>",pos)
    print("componud>>>",compound)
    

    print ( "graph_calculation_list" , graph_calculation_list) 


    data = {
        
            "result_all_plot":result_final[3], 
            
            "emotional_rollercoaster":round(emotional_rollercoaster,2), 
            "plot_complexity":round(plot_complexity,2), 
            "degree_conflict": round(degree_conflict,2), 
            
            "result_plot_complexity" : result_final[0],
            "result_emotional_rollercoaster": result_final[1],
            "result_degree_conflict" : result_final[2],
            
            "neg" : neg,
            "neu" : neu,
            "pos" : pos,
            "compound" : compound,
            "graph_calculation_list" : graph_calculation_list,
            "Shifts Between Positive and Negative Sentiments" :plot_conf_re[9]
        }
    
    
    
    return data 






###### 실행 테스트  ######
print("result: ",ai_plot_coflict_total_analysis(input_text))


### {'result_all_plot': 1.33, 'emotional_rollercoaster': 25.0, 'plot_complexity': 26.52, 'degree_conflict': 3.0, 'result_plot_complexity': 0, 'result_emotional_rollercoaster': 0, 'result_degree_conflict': 2, 'neg': [0.0, 0.0, 0.041, 0.044, 0.19, 0.0, 0.04, 0.0, 0.054, 0.0, 0.0, 0.118, 0.101, 0.0, 0.239, 0.133, 0.104, 0.0, 0.083, 0.09, 0.092, 0.058, 0.079, 0.121], 'neu': [0.808, 1.0, 0.778, 0.787, 0.678, 1.0, 0.884, 1.0, 0.79, 0.723, 1.0, 0.882, 0.739, 1.0, 0.761, 0.867, 0.896, 0.762, 0.702, 0.77, 0.667, 0.822, 0.702, 0.65], 'pos': [0.192, 0.0, 0.181, 0.169, 0.132, 0.0, 0.076, 0.0, 0.155, 0.277, 0.0, 0.0, 0.161, 0.0, 0.0, 0.0, 0.0, 0.238, 0.215, 0.14, 0.242, 0.119, 0.219, 0.23], 'compound': [0.228, 0.0, 0.7269, 0.6486, -0.2144, 0.0, 0.4588, 0.0, 0.624, 0.7579, 0.0, -0.5267, 0.0258, 0.0, -0.5719, -0.3354, -0.2732, 0.6486, 0.4588, 0.2206, 0.7351, 0.3182, 0.4939, 0.5719], 'graph_calculation_list': [0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -2.6, -2.7, -2.8, -2.9, -3.0, -3.1, -3.2, -3.3, -3.4, -3.5, -3.6, -3.7, -3.8, -3.9, -4.0, -4.1, -4.2, -4.3, -4.4, -4.5, -4.6, -4.7, -4.8, -4.9, -5.0, -5.1, -5.2, -5.3, -5.4, -5.5, -5.6, -5.7, -5.8, -5.9, -6.0, -6.1, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7, -6.8, -6.9, -7.0, -7.1, -7.2, -7.3, -7.4, -7.5, -5.5, -5.6, -5.7, -5.8, -3.8, -3.9, -4.0, -4.1, -4.2, -4.3, -2.3, -2.4, -2.5, -2.6, -2.7, -2.8, -2.9, -3.0, -3.1, -3.2, -3.3, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -2.6, -2.7, -2.8, -2.9, -3.0, -3.1, -3.2, -3.3, -3.4, -3.5, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -2.6, -2.7, -2.8, -2.9, -3.0, -3.1, -3.2, -3.3, -3.4, -3.5, -3.6, -3.7, -3.8, -3.9, -4.0, -4.1, -4.2, -4.3, -4.4, -4.5, -4.6, -4.7, -4.8, -4.9, -5.0, -5.1, -5.2, -5.3, -3.3, -3.4, -3.5, -3.6, -3.7, -3.8, -3.9, -4.0, -4.1, -4.2, -4.3, -4.4, -4.5, -4.6, -4.7, -4.8, -4.9, -5.0, -5.1, -5.2, -5.3, -5.4, -5.5, -5.6, -5.7, -5.8, -5.9, -6.0, -6.1, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7, -6.8, -6.9, -7.0, -7.1, -7.2, -7.3, -7.4, -7.5, -7.6, -7.7, -7.8, -7.9, -8.0, -8.1, -8.2, -8.3, -8.4, -8.5, -8.6, -8.7, -8.8, -8.9, -6.9, -7.0, -7.1, -5.1, -5.2, -5.3, -5.4, -5.5, -5.6, -5.7, -5.8, -5.9, -6.0, -6.1, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7, -4.7, -4.8, -4.9, -5.0, -5.1, -5.2, -5.3, -5.4, -5.5, -5.6, -5.7, -5.8, -5.9, -6.0, -6.1, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7, -6.8, -6.9, -7.0, -7.1, -7.2, -7.3, -7.4, -7.5, -7.6, -7.7, -7.8, -7.9, -8.0, -6.0, -6.1, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7, -6.8, -6.9, -7.0, -7.1, -7.2, -7.3, -7.4, -7.5, -7.6, -7.7, -7.8, -7.9, -8.0, -8.1, -8.2, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7, -6.8, -6.9, -7.0, -7.1, -7.2, -7.3, -7.4, -7.5, -7.6, -7.7, -7.8, -7.9, -8.0, -8.1, -8.2, -8.3, -8.4, -8.5, -8.6, -8.7, -8.8, -8.9, -9.0, -9.1, -9.2, -9.3, -9.4, -9.5, -9.6, -9.7, -9.8, -9.9, -10.0, -10.1, -10.2, -10.3, -10.4, -10.5, -10.6, -10.7, -10.8, -10.9, -11.0, -11.1, -11.2, -11.3, -11.4, -11.5, -11.6, -11.7, -11.8, -11.9, -12.0, -12.1, -12.2, -12.3, -12.4, -12.5, -12.6, -12.7, -12.8, -12.9, -13.0, -13.1, -11.1, -11.2, -11.3, -11.4, -11.5, -11.6, -11.7, -11.8, -11.9, -12.0, -12.1, -12.2, -12.3, -12.4, -12.5, -12.6, -12.7, -12.8, -12.9, -13.0, -13.1, -13.2, -13.3, -13.4, -13.5, -13.6, -13.7, -13.8, -13.9, -14.0, -14.1, -14.2, -14.3, -14.4, -14.5, -14.6, -14.7, -14.8, -14.9, -15.0, -15.1, -15.2, -15.3, -13.3, -13.4, -13.5, -13.6, -13.7, -13.8, -13.9, -14.0, -14.1, -14.2, -14.3, -14.4, -14.5, -14.6, -14.7, -14.8, -14.9, -15.0, -13.0, -13.1, -13.2, -13.3, -13.4, -13.5, -13.6, -13.7, -13.8, -13.9, -14.0, -14.1, -14.2, -14.3, -14.4, -12.4, -12.5, -10.5, -10.6, -10.7, -10.8, -10.9, -11.0, -11.1, -11.2, -11.3, -11.4, -11.5, -11.6, -11.7, -11.8, -11.9, -9.9, -10.0, -10.1, -10.2, -10.3, -10.4, -10.5, -10.6, -8.6, -8.7, -8.8, -8.9, -9.0, -9.1, -9.2, -9.3, -9.4, -9.5, -9.6, -9.7, -9.8, -9.9, -10.0, -10.1, -10.2, -10.3, -10.4, -8.4, -8.5, -8.6, -8.7, -8.8, -6.8, -6.9, -7.0, -7.1, -7.2, -7.3, -7.4, -7.5, -7.6, -7.7, -7.8, -7.9, -8.0, -8.1, -8.2, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7, -6.8, -6.9, -4.9, -5.0, -5.1, -5.2, -5.3, -5.4, -5.5, -5.6, -5.7, -5.8, -5.9, -6.0, -6.1, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7, -6.8, -6.9, -7.0, -7.1, -7.2, -7.3, -7.4, -7.5, -7.6, -7.7, -7.8, -7.9, -5.9, -6.0, -6.1, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7, -6.8, -6.9, -7.0, -7.1, -7.2, -7.3, -7.4, -7.5, -7.6, -7.7, -7.8, -7.9, -8.0, -8.1, -8.2, -8.3, -8.4, -8.5, -8.6, -8.7, -8.8, -8.9, -9.0, -9.1, -9.2, -9.3, -9.4, -9.5, -7.5, -7.6, -7.7, -7.8, -7.9, -8.0, -8.1, -8.2, -8.3, -8.4, -8.5, -8.6, -8.7, -8.8, -8.9, -9.0, -9.1, -9.2, -9.3, -9.4, -9.5, -7.5, -7.6, -7.7, -7.8, -7.9, -8.0, -6.0, -6.1, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7, -6.8]}




# 1.ai_plot_coflict_total_analysis(input_text) 실행하면, 

# 2.결과 나옴!(그래프 2개, 적합성, 복잡성 등등 값 도출됨)

# ACTION VERBS RATIO : 8.537
# ====================================================================
# 에세이에 표현된 다양한 감정 수: 7
# ====================================================================
# 문장에 표현된 감정 비율 :  25.0
# ====================================================================
# ['before', 'above', 'sound', 'trail', 'by', 'way', 'until', 'city', 'from', 'to', 'after', 'against', 'forge', 'on', 'path', 'view', 'during', 'through', 'in', 'up', 'camp', 'route']
# ====================================================================
# SETTING RATIO :  12.34
# ====================================================================
# 전체 문장에서 캐릭터를 의미하는 단어나 유사어 비율 : 8.79
# Degree of conflict  단어가 전체 문장(단어)에서 차지하는 비율 계산 : 3.0
# 감정기복비율 : 25.0
# 셋팅비율 계산 :  12.34
# Plot Complxity : 26.517731803455586
# ===============================================================================
# ======================      Degree of Conflict   ==============================
# ===============================================================================
# 1명의 에세이 결과 계산점수 : (26.517731803455586, 25.0, 3.0)
# min_ 32
# max_:  128
# div_: 32
# cal_abs 절대값 : 53.48226819654441
# compare7 : 17.75295530057593
# compare6 : 21.30354636069112
# compare5 : 26.629432950863897
# compare4 : 35.50591060115186
# compare3 : 53.258865901727795
# Lacking: 2
# min_ 25
# max_:  102
# div_: 25
# cal_abs 절대값 : 39.0
# compare7 : 14.833333333333334
# compare6 : 17.8
# compare5 : 22.25
# compare4 : 29.666666666666668
# compare3 : 44.5
# Lacking: 2
# min_ 0
# max_:  0
# div_: 0
# cal_abs 절대값 : 2.686
# compare7 : 0.5523333333333333
# compare6 : 0.6628000000000001
# compare5 : 0.8285
# compare4 : 1.1046666666666667
# compare3 : 1.657
# Overboard: 2
# fin_result: [(0, 1), (0, 2), (2, 1)]

# result

#  [0, 0, 2, 1.33, 5.0, 3.0, 26.52]


# select prompt number to get intended mood

def intended_mood_by_prompt(promptNo):
    if promptNo == 'prompt_1':
        intended_mood = ['joy', 'pride', 'approval']
    elif promptNo == "prompt_2":
        intended_mood = ['disappointment', 'fear', 'confusion']
    elif promptNo == "prompt_3":
        intended_mood = ['curiosity', 'disapproval', 'realization']
    elif promptNo == "prompt_4":
        intended_mood = ['gratitude', 'surprise', 'admiration']
    elif promptNo == "prompt_5":
        intended_mood = ['realization', 'pride', 'admiration']
    elif promptNo == "prompt_6":
        intended_mood = ['curiosity', 'excitement', 'confusion']
    elif promptNo == "prompt_7":
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
    top5Emotions.append(top5Emo[0][0])
    top5Emotions.append(top5Emo[1][0])
    top5Emotions.append(top5Emo[2][0])
    top5Emotions.append(top5Emo[3][0])
    top5Emotions.append(top5Emo[4][0])
    
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
            re_mood = "suspensefull"
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
        sentence1 = ['You’ve intended to write the essay in a disturbed mood.']
        sentence2 = ['The AI’s analysis shows that your personal statement’s mood seems to be disturbed.']

    elif re_mood == 'suspenseful':
        sentence1 = ['You’ve intended to write the essay in a suspenseful mood.']
        sentence2 = ['The AI’s analysis shows that your personal statement’s mood seems to be suspenseful.']

    elif re_mood == 'sad':
        sentence1 = ['You’ve intended to write the essay in a sad mood.']
        sentence2 = ['The AI’s analysis shows that your personal statement’s mood seems to be sad.']

    elif re_mood == 'joyful':
        sentence1 = ['You’ve intended to write the essay in a joyful mood.']
        sentence2 = ['The AI’s analysis shows that your personal statement’s mood seems to be joyful.']
                     
    elif re_mood == 'calm':
        sentence1 = ['You’ve intended to write the essay in a calm mood.']
        sentence2 = ['The AI’s analysis shows that your personal statement’s mood seems to be calm.']

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
    
    if promt_number == 'prompt_1': # 1번 문항을 선택했을 경우(문항선택 'prompt_1 ~ 7')
        accepted_essay_av_value = prompt_1_sent_mean
        
    elif promt_number == 'prompt_2':
        accepted_essay_av_value = prompt_2_sent_mean
        
    elif promt_number == 'prompt_3':
        accepted_essay_av_value = prompt_3_sent_mean
        
    elif promt_number == 'prompt_4':
        accepted_essay_av_value = prompt_4_sent_mean
        
    elif promt_number == 'prompt_5':
        accepted_essay_av_value = accepted_essay_av_value = prompt_5_sent_mean
        
    elif promt_number == 'prompt_6':
        accepted_essay_av_value = prompt_6_sent_mean
        
    elif promt_number == 'prompt_7':
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
 
    print(ps_ext_emo)
    
    group_ext_emo = [] # 그룹 에세이에서 추출한 5개의 평균 대표감성 5개
    for item_2 in accepted_essay_av_value:
        group_ext_emo.append(item_2[0])
    
    print(group_ext_emo)
    
    #두 값을 비교하여 very close / somewhat close / weak 결정
    #중복요소를 추출하여 카운팅하면 두 총 리스트의 값 중에서 중복요소가 몇개 있는지 알 수 있을때 유사도를 계산할 수 있음
    count={}
    sum_emo = ps_ext_emo + group_ext_emo
    for m in sum_emo:
        try: count[m] += 1
        except: count[m] = 1
    print('중복값:', count)
    
    compare_re = []
    for value in count.values(): # 딕셔너리의 벨류 값을 하나씩 가져와서 
        if value > 1: # 1보다 큰 수는 중복된 수 이기 때문에 
            compare_re.append(value) # 중복된 수를 새로운 리스트 compare_re에 넣고
        else:
            pass
        
    sum_compare_re = sum(compare_re) 
    # 리스트의 숫자를 모두 더해서 최종 비교를 할거임,
    # 총 리스틔 수는 10개이고 중복 최대값은 5개 모두가 중복되는 10이고 최소값은 0(아무것도 중복되지 않음)   0~10까지의 수로 표현됨
    print(sum_compare_re)
    
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
    # 5.accepted_essay_av_value : 1000명의 합격한 학생의 대표감서 5개
    # 6.in_depth_sent_result : 최종 심층 분석결과
    # 7.re_mood : 개인 mood 분석 추출결과

    return result_emo_list, intendedMoodByPmt, detected_mood, sentence1, sentence2, sentence3, key_emo[:5], accepted_essay_av_value, in_depth_sent_result, re_mood
                    
                    

# Shifts Between Positive and Negative Sentiments 문장 생성 부분
def shifts_Bt_PoNe(prompt_no, input_val):
    group_pmt_no_1 = 20
    group_pmt_no_2 = 20
    group_pmt_no_3 = 20
    group_pmt_no_4 = 20
    group_pmt_no_5 = 20
    group_pmt_no_6 = 20
    group_pmt_no_7 = 20

    
    sentense_1_PN = ['The admitted cases tend to display', , 'fluctuations between polarizing sentiments throughout the story for this type of essay prompt.']




def feedback_plot_conflict(prompt_no, ps_input_text):
    
    #############################################
    # 합격한 학생의 평균 mood 평균 값
    # gruop_input_text_re : 합격한 학생의 평균값
    group_input_text_re = 'Joyful'
    
    # intended mood
    int_mood_by_prompt_re = intended_mood_by_prompt(prompt_no)
    
    # detected mood
    dtc = ai_emotion_analysis(ps_input_text, prompt_no)
    intended_mood = dtc[1] # intended Mood 
    print('intended_mood:', intended_mood)
    detected_mood = dtc[2] # detected Mood
    print('detected_mood:', detected_mood)
    
    # print("추출된 감성 5개: ", dtc[7])
    
    
    # intended value
    if intended_mood == ['disturbed']:
        pc_intended_mood_result = 'Disturbed'
        pc_tension = 'Moderate & High Tension'
    elif intended_mood == ['suspenseful']:
        pc_intended_mood_result = 'Suspenseful'
        pc_tension = 'High Tension'
    elif intended_mood == ['sad']:
        pc_intended_mood_result = 'Sad'
        pc_tension = 'Moderate Tension'
    elif intended_mood == ['joy']:
        pc_intended_mood_result = 'Joyful'
        pc_tension = 'Moderate & Low Tension'
    else: # carm
        pc_intended_mood_result = 'Carm'
        pc_tension = 'Low Tension'
        
    # Tension Value 계산하기 'Moderate & High Tension', 'High Tension', 'Moderate Tension', 'Moderate & Low Tension', 'Low Tension'
    if detected_mood == ['disturbed']:
        dtc_intended_mood_result = 'Disturbed'
        dtc_tension = 'Moderate & High Tension'
    elif detected_mood == ['suspenseful']:
        dtc_intended_mood_result = 'Suspenseful'
        dtc_tension = 'High Tension'
    elif detected_mood == ['sad']:
        dtc_intended_mood_result = 'Sad'
        dtc_tension = 'Moderate Tension'
    elif detected_mood == ['joy']:
        dtc_intended_mood_result = 'Joyful'
        dtc_tension = 'Moderate & Low Tension'
    else: # carm
        dtc_intended_mood_result = 'Carm'
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
    plot_conf_re = ai_plot_conf(input_text)
    
    #####################################################
    ################# 합격한 학생의 평균값 ###################
    group_conflict_word_num = 5 # Conflict words numbers
    group_action_verbs_num = 20 # 
    
    count_conflict_list_re = plot_conf_re[5] # conflict words list
    print('count_conflict_list_re', count_conflict_list_re)
    nums_conflict_words_re = plot_conf_re[6] # conflict words number
    print('nums_conflict_words_re', nums_conflict_words_re)
    get_words__re = plot_conf_re[7] # Action Verbs list
    print('get_words__re', get_words__re)
    nums_action_verbs_re = plot_conf_re[8] # Action verbs number
    print('nums_action_verbs_re', nums_action_verbs_re)
    
    # 개인, 그룹의 결과 비교,  컨플릭단어와 액션동사 사용량 비교
    if nums_conflict_words_re == group_conflict_word_num and nums_action_verbs_re == group_action_verbs_num:
        stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', 'similar number of', 'conflict oriented words and', 'similar number of', 'action verbs in your story.']
        stm_sentence_2 = ['Overall, your plot's tension level looks good compared with the accepted average.']
    elif nums_conflict_words_re > (group_conflict_word_num - group_conflict_word_num * 0.3)  and nums_conflict_words_re < (group_conflict_word_num + group_conflict_word_num * 0.3): 
                                                                                                                
        if nums_action_verbs_re > (group_action_verbs_num - group_action_verbs_num * 0.3) and nums_action_verbs_re < (group_action_verbs_num + group_action_verbs_num * 0.3):
            stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', 'similar number of', 'conflict oriented words and', 'similar number of', 'action verbs in your story.']
            stm_sentence_2 = ['Overall, your plot's tension level looks good compared with the accepted average.']
        else:
            pass
                                                                                                                    
    elif nums_conflict_words_re < (group_conflict_word_num + group_conflict_word_num * 0.3)  and nums_action_verbs_re < (group_action_verbs_num + group_action_verbs_num * 0.3):
        
        if nums_action_verbs_re < (group_action_verbs_num + group_action_verbs_num * 0.3) and nums_action_verbs_re > (group_action_verbs_num - group_action_verbs_num * 0.3):                                                                                                           
            stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', 'similar number of', 'conflict oriented words and', 'similar number of', 'action verbs in your story.']
            stm_sentence_2 = ['Overall, your plot's tension level looks good compared with the accepted average.']
        else:
            pass

    elif nums_conflict_words_re > group_conflict_word_num and nums_action_verbs_re > group_action_verbs_num:
        first_value = abs(nums_conflict_words_re - group_conflict_word_num)
        second_value = abs(nums_action_verbs_re - group_action_verbs_num)
        stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', first_value, 'more', 'conflict oriented words and', second_value,'more', 'action verbs in your story.']
        stm_sentence_2 = ['Overall, you may consider', 'using less' , 'words to', 'alleviate', 'the plot's tension]

    elif nums_conflict_words_re > group_conflict_word_num and nums_action_verbs_re < group_action_verbs_num:
        first_value = abs(nums_conflict_words_re - group_conflict_word_num)
        second_value = abs(nums_action_verbs_re - group_action_verbs_num)
        stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', first_value, 'more', 'conflict oriented words and', second_value,'fewer', 'action verbs in your story.']
        stm_sentence_2 = ['Overall, you may consider', 'using less' , 'words to', 'intensify', 'the plot's tension]

    elif nums_conflict_words_re < group_conflict_word_num and nums_action_verbs_re > group_action_verbs_num:
        first_value = abs(nums_conflict_words_re - group_conflict_word_num)
        second_value = abs(nums_action_verbs_re - group_action_verbs_num)
        stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', first_value, 'fewer', 'conflict oriented words and', second_value,'more', 'action verbs in your story.']
        stm_sentence_2 = ['Overall, you may consider', 'adding more' , 'words to', 'alleviate', 'the plot's tension]

    elif nums_conflict_words_re < group_conflict_word_num and nums_action_verbs_re < group_action_verbs_num:
        first_value = abs(nums_conflict_words_re - group_conflict_word_num)
        second_value = abs(nums_action_verbs_re - group_action_verbs_num)
        stm_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', first_value, 'fewer', 'conflict oriented words and', second_value,'fewer', 'action verbs in your story.']
        stm_sentence_2 = ['Overall, you may consider', 'adding more' , 'words to', 'intensify', 'the plot's tension]
    else:
        pass
    
    # Shifts Between Positive and Negative Sentiments 문장 생성 부분 (def shifts_Bt_PoNe(input_val))



    # 결과해석 #
    # 1.pc_intended_mood_result : 개인이 선택한 intended mood 선택 결과
    # 2.pc_tension : 개인의 선택한 tension 결과
    
    # 3.intended_mood : 개인의 에세이를 prompt 문항에 의해 분석한 결과 ---> 이 값을 적용할 것
    # 4.detected_mood : 개인의 에세이를 감성분석한 결과
    # 5.comp_int_dtc : Intended Mood vs. Plot & Conflict 부분의 두 값 비교로 같거(=)나 같지 않음(!=)
    # 6.sentence_1 ~ 2 : 문장생성
    # 7.stm_sentence_1 : number of stimulus words
        
    return pc_intended_mood_result, pc_tension, intended_mood, detected_mood, comp_int_dtc, sentence_1, sentence_2, stm_sentence_1, stm_sentence_2


### 실행 ###
#  - 입력값 - #
prompt_no = 'prompt_1' #이런 형식으로 넣어야 함
ps_input_text = input_text # 위에서 이미 입력

result_feedback = feedback_plot_conflict(prompt_no, ps_input_text)

print('resut:', result_feedback)