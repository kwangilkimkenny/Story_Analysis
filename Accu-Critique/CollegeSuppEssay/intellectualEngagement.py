# Major Fit 20%
# General Academic Topics 30%
# Prompt Oriented Sentiment 20%
# Intellectual Engagement 20%(Cohesion level 40% + Academic Verbs 60%)

# ohesion level >>> 계산 완료

# 만약 특정 coherence가 너무 높아지면 정보의 양이 줄어들게 되고, coherence가 너무 낮아 정보들이 연관성이 없다면, 분석의 의미가 낮아짐 --- 적당해야 함... 적당?? 

# ref: https://docs.google.com/document/d/1P0EP7-T4AL-btgIrOPUOeSaQ8qOmcoRpMEZXIgIcOBU/edit


import re
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk import pos_tag
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from tqdm import tqdm

# home : py37Keras

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

from gensim.corpora import Dictionary



# input College Supp Essay 
essay_input = """I inhale deeply and blow harder than I thought possible, pushing the tiny ember from its resting place on the candle out into the air. The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""



essay_input_corpus = str(essay_input) #문장입력
essay_input_corpus = essay_input_corpus.lower()#소문자 변환
print('essay_input_corpus :', essay_input_corpus)

sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화 > 문장으로 구분
total_sentences = len(sentences)#토큰으로 처리된 총 문장 수
total_words = len(word_tokenize(essay_input_corpus))# 총 단어수
print(total_words)
split_sentences = []
for sentence in sentences:
    processed = re.sub("[^a-zA-Z]"," ", sentence)
    words = processed.split()
    split_sentences.append(words)


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


# 역토큰화
detokenized = []
for i_token in range(len(preprossed_sent_all)):
    for tk in preprossed_sent_all:
        t = ' '.join(tk)
        detokenized.append(t)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=None)
X = vectorizer.fit_transform(detokenized)

from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=777, max_iter=1)
lda_top = lda_model.fit_transform(X)
print(lda_model.components_)
print(lda_model.components_.shape)

terms = vectorizer.get_feature_names() 
# 단어 집합. 1,000개의 단어가 저장되어있음.
topics_ext = []
def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        #print("Topic %d :" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n -1:-1]])
        topics_ext.append([(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n -1:-1]])
        


topics_ext.append(get_topics(lda_model.components_, terms))

topics_ext = list(filter(None, topics_ext))

result_ = []
cnt = 0
cnt_ = 0
for ittm in range(len(topics_ext)):
    #print('ittm:', ittm)
    cnt_ = 0
    for t in range(len(topics_ext[ittm])-1):
        
        #print('t:', t)
        add = topics_ext[ittm][cnt_][0]
        result_.append(add)
        #print('result_:', result_)
        cnt_ += 1


topics = list(set(result_))
print("=== topics ==== :", topics) # === topics ==== : ['india', 'love', 'shoot', 'wish', 'different', ....

# 입력문장에서 추출한 최종 키워드들(중복값을 제거한 값) -- 하지만 아래서 한번 더 처리할거임
print('Ext Topic words: ',  topics)


# dictionary 만들기
def make_dictionary(essay_input_data):
    # tokenize sentence
    sent_token_re = sent_tokenize(essay_input_data)
    processed_data_re = []
    for i in sent_token_re:
        tokens_re = nltk.word_tokenize(i)
        # stopwords 제거, 각 문장을 소문자로 변환하고, 명사, 동사만 추출, 문장별로 리스트 만들어서 리스트에 다시 담기
        stop = set(stopwords.words('english'))
        clean_tokens = [tok for tok in tokens_re if len(tok.lower())>1 and (tok.lower() not in stop)]
        tagged = nltk.pos_tag(clean_tokens)
        print('tagged:', tagged)
        allnoun = [word for word, pos in tagged if pos in ['NN','NNP','VB']]
        processed_data_re.append(allnoun)

    return processed_data_re


processed_data =  make_dictionary(essay_input)
print("=====================================")
print('processed_data :', processed_data)

# 정수 인코딩과 빈도수 생성 
dictionary = corpora.Dictionary(processed_data)
print('dictionary:', dictionary) # dictionary: Dictionary(167 unique tokens: ['air', 'candle', 'ember', 'harder', 'place']...)

#딕셔너리에서 빈도수 작은 값은 제거한다. 너무 많이 나오는 값도 제거한다.
dictionary.filter_extremes(no_below=2, no_above=0.05) 
corpus = [dictionary.doc2bow(text) for text in processed_data] 
print('corpus:', corpus)

# processed_data의 이중 리스트를 flatten하게 만들고, 여기서 앞서 추출한 Topic Keywords와 비교하여 겹치는 것만 추출하고, 이것의 Coherence Value를 계산하면 됨
flatten_dic_list = [y for x in processed_data for y in x]
#print("=====================================")
#print('flatten_dic_list:', flatten_dic_list)

# 비교하기
ext_key_topic_for_cal_cohesion_value_list = [] # ----> 이 값이 최종 비교 토픽들임
for tp in topics:
    if tp in flatten_dic_list:
        ext_key_topic_for_cal_cohesion_value_list.append(tp)

# 이 값을 가지고 응집성(Cohesion) 계산
ext_compare_topics = ext_key_topic_for_cal_cohesion_value_list
print('+++++ ext_compare_topics +++++', ext_compare_topics)

# 여기서 딕셔너리의 값과 다시 비교하여, 딕셔너리에 있는 값에서 topic keywords를 추출해야 함

# 딕셔너리를 리스트로 변환
dic_tuple_list = list(zip(dictionary.keys(),dictionary.values()))
#print('dict_tuple_list:', dic_tuple_list)
# 딕셔너리의 value만 리스트로 변환 ---> 이 값을 다시 ext_compare_topics 와 비교하여 겹치는 것만 계산한다.
dict_value_list = list(dictionary.values())
print('dict_value_list:', dict_value_list)

fin_topic_re = [] #------> 최종적으로 이 값이 coherence value를 계산하는 단어들임
for ttpitm in ext_compare_topics:
    if ttpitm in dict_value_list:
        fin_topic_re.append(ttpitm)


final_topics = [fin_topic_re]
print('final_topics:', final_topics)

# coherence value
cm = CoherenceModel(topics=final_topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
coherence = cm.get_coherence()

print('Choerence Value:', coherence)

# Choerence Value: -20.458978939301947  이렇게 추출됨, 의미는 높으면 일관성있게 문장을 작성했다는 의미, 낮으면 내용이 산만하다는 의미