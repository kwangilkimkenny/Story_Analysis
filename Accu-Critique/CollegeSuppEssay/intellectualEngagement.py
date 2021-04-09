# Major Fit 20%
# General Academic Topics 30%
# Prompt Oriented Sentiment 20%
# Intellectual Engagement 20%(Cohesion level 40% + Academic Verbs 60%)
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

# 입력문장에서 추출한 최종 키워드들(중복값을 제거한 값)
print('Ext Topic words: ',  topics)

topics_ = [
    ['wish', 'reason', 'excite', 'mridangam', 'support']
]




# dictionary 만들기
def clean_text(text):
    #영문, 숫자만 남기고 제거한다. :param text: :return: 
    text = text.replace(".", " ").strip()
    text = text.replace("·", " ").strip() 
    pattern = '[^|0-9|a-zA-Z]+' 
    text = re.sub(pattern=pattern, repl='', string=text) 
    return text


def get_nouns(sentence): 
    tagged = nltk.pos_tag(sentence) 
    nouns = [s for s, t in tagged if t in ['SL', 'NNG', 'NNP'] and len(s) > 1] 
    return nouns


def tokenize(txt): 
    tokenizer = word_tokenize
    processed_data = [] 
    for sent in tqdm(txt): 
        sentence = clean_text(sent.replace('\n', '').strip()) 
        processed_data.append(get_nouns(sentence)) 
    return processed_data


processed_data = tokenize(sentences)
print('processed_data:', processed_data)



#참고 : https://www.lucypark.kr/courses/2015-dm/text-mining.html
#참고 : https://joyhong.tistory.com/138

####### processed_data 가 이상하게 나옴 이 부분 수정할 것!!!!




# 정수 인코딩과 빈도수 생성 
dictionary = corpora.Dictionary(processed_data)
print('dictionary:', dictionary)

dictionary.filter_extremes(no_below=10, no_above=0.05)
corpus = [dictionary.doc2bow(text) for text in processed_data]



# coherence value
cm = CoherenceModel(topics=topics_, corpus=corpus, dictionary=dictionary, coherence='u_mass')
coherence = cm.get_coherence()

print('Choerence Value:', coherence)