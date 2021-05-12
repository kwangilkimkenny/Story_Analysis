# 1)  Prompt Oriented Keywords (20%)
# achievement 와 pride / proud 등의 단어들과 에세이의 다른 key topic들이 가까이 있는지 아닌지
# 유사의/유의어 + vector 함께 고려

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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop = stopwords.words('english')


# 토픽 추출
def getTopics(essay_input):

    # step_1
    essay_input_corpus = str(essay_input) #문장입력
    essay_input_corpus = essay_input_corpus.lower()#소문자 변환
    #print('essay_input_corpus :', essay_input_corpus)

    sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화 > 문장으로 구분
    total_sentences = len(sentences)#토큰으로 처리된 총 문장 수
    total_words = len(word_tokenize(essay_input_corpus))# 총 단어수
    #print(total_words)
    split_sentences = []
    for sentence in sentences:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences.append(words)

    # step_2
    lemmatizer = WordNetLemmatizer()
    preprossed_sent_all = []
    for i in split_sentences:
        preprossed_sent = []
        for i_ in i:
            if i_ not in stop: #remove stopword
                lema_re = lemmatizer.lemmatize(i_, pos='v') #표제어 추출, 동사는 현재형으로 변환, 3인칭 단수는 1인칭으로 변환
                if len(lema_re) > 3: # 단어 길이가 3 초과단어만 저장(길이가 3 이하는 제거)
                    preprossed_sent.append(lema_re)
        preprossed_sent_all.append(preprossed_sent)

    # step_3 : 역토큰화로 추출된 단어를 문자별 단어리스트로 재구성
    # 역토큰화
    detokenized = []
    for i_token in range(len(preprossed_sent_all)):
        for tk in preprossed_sent_all:
            t = ' '.join(tk)
            detokenized.append(t)

    # step_4 : TF-IDF 행렬로 변환
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=None)
    X = vectorizer.fit_transform(detokenized)

    from sklearn.decomposition import LatentDirichletAllocation
    lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=777, max_iter=1)
    lda_top = lda_model.fit_transform(X)
    # print(lda_model.components_)
    # print(lda_model.components_.shape)


    terms = vectorizer.get_feature_names() 
    # 단어 집합. 1,000개의 단어가 저장되어있음.
    topics_ext = []
    def get_topics(components, feature_names, n=5):
        for idx, topic in enumerate(components):
            print("Topic %d :" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n -1:-1]])
            topics_ext.append([(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n -1:-1]])
    # 토픽 추출 시작
    topics_ext.append(get_topics(lda_model.components_, terms))
    # 토픽중 없는 값 제거 필터링
    topics_ext = list(filter(None, topics_ext))


    # 최종 토필 추축
    result_ = []
    cnt = 0
    cnt_ = 0
    for ittm in range(len(topics_ext)):
        print('ittm:', ittm)
        cnt_ = 0
        for t in range(len(topics_ext[ittm])-1):
            
            print('t:', t)
            add = topics_ext[ittm][cnt_][0]
            result_.append(add)
            print('result_:', result_)
            cnt_ += 1

    #추출된 토픽들의 중복값 제거
    result_fin = list(set(result_))

    #### 토픽 추출한 결과 반환 ####
    return result_fin


## 연관어 불어오기 ##
#유사단어를 추출하여 리스트로 반환
def get_sim_words(text):

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

    #비교분석 단어들을 리스트에 넣어서 필터로 만들고
    words_list = ['api','expression','abrasion','underrepresented','racism','zero','bame','diversity','ableism','bme','diaspora','heritage','intersex','resource','sex','group','multiracial','orientation','ageism','socioeconomic','butch','equity','affirmative','attribution','mixed','psychological','simply','bias','creative','ally','non-binary','dei','asexual','affinity','dual','dominant','sexual','white','accessibility','allyship','neurodiverse','employee','straight','imposter','privilege','black','tax','transphobia','reassignment','homosexual','bi-cultural','prejudice','biphobia','genderqueer','mixed-ethnicity','innate','microadvantages','inclusion','fit','gsd','queer','“mixed”','identity','supremacy','aapi','gay','cognitive','discrimination','lgbtqi','ethnocentrism','xenophobia','transitioning','lesbophobia','oppression','bi','outgroup','transgender','confirmation','threat','ethnic','error','responsibility','sponsor','game','ace','mansplain','transsexual','social','behavioral','culture','dysphoria','people','trans','group','cover','groupthink','stereotypes','homophobia','conscious','pronoun','pan','sum','stereotype','cis','safety','equality','bias','inclusion','intersectionality','microaggression','deadnaming','disability','lesbian','microaffirmations','in-group','privilege','corporate','unconscious','atheism','heterosexual','lgbtqia','diversity','hepeating','gender','inclusive','poc','syndrome','lgbtq+','action','leader','mixed-race','mentor','–','emotional','femme','workplace','prejudice','cisgender','color']
    
    ####문장에 list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
    
    #우선 토큰화한다.
    retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
    token_input_text = retokenize.tokenize(essay_input_corpus)
    #print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
    #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
    filtered_chr_text = []
    for k in token_input_text:
        for j in words_list:
            if k == j:
                filtered_chr_text.append(j)
    
    #print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
    
    filtered_chr_text_ = set(filtered_chr_text) #중복제거
    filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환

    ext_sim_words_key = filtered_chr_text__

    return ext_sim_words_key


