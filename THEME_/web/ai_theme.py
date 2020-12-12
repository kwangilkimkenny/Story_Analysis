# 이 코드는 테마를 분석하는 기능도 있지만, 글의 가독성을 모두 추출 분석하는 기능을 가지고 있다.
# def pronouns_Anaphoric(text): >>> 전체 글에서 얼마나 포함되어 있는지 비율 도출
# def qanda_analysis(input_text): >>> 질문에 적절한 답을 제시해주는지 분석해준다.
# def connecitve_words_ratio(text): >>> 연결어에 대한 전체 문장당 포함 비율을 도출해준다.
# results = readability.getmeasures(text_input, lang='en') >>> 가독성 등 모든 것을 분석해준다.

# 종필 : 이 결과를 홈페이지에서 작동하도록 구현해줘. 

# 해야할 일: 분석 결과를 그래프로 표현할 때 에세이 분석 알고리즘을 구현하여 분석보고서를 작성할 수 있도록 구현해야 함(KJ논리적용)


#conflict
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
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
from gensim import corpora, models, similarities

import readability


##################################################################



#Anaphoric reference(지시대명사)

def pronouns_Anaphoric(text):

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

    #Anaphoric reference 단어들을 리스트에 넣어서 필터로 만들고
    character_list = ['all','another','any','anybody','anyone','anything','as','aught','both','each'
                    ,'each other','either','enough','everybody','everyone','everything','few','he','her','hers','herself','him','his','I'
                    ,'idem','it','its','itself','many','me','mine','most','my','myself','naught','neither','no one','nobody','none'
                    ,'nothing','nought','one','one another','other','others','ought','our','ours','ourself','ourselves','several'
                    ,'she','some','somebody','someone','something','somewhat','such','suchlike','that','thee','their','theirs'
                    ,'theirself','theirselves','them','themself','themselves','there','these','they','thine','this','those','thou'
                    ,'thy','thyself','us','we','what','whatever','whatnot','whatsoever','whence','where','whereby','wherefrom','wherein'
                    ,'whereinto','whereof','whereon','wherever','wheresoever','whereto','whereunto','wherewith','wherewithal','whether'
                    ,'which','whichever','whichsoever','who','whoever','whom','whomever','whomso','whomsoever','whose','whosever','whosesoever'
                    ,'whoso','whosoever','ye','yon','yonder','you','your','yours','yourself','yourselves']
    
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
        ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
    
    pronouns_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 캐릭터 표현 수
    pronouns_count_ = len(filtered_chr_text__) #중복제거된 캐릭터 표현 총 수
        
    result_pronouns_ratio = round(pronouns_total_count/total_words * 100, 2)

    #return result_pronouns_ratio, total_sentences, total_words, pronouns_total_count, char_count_, ext_sim_words_key
    return result_pronouns_ratio





#선택한 질문과 비교해서 결과가 0.9 이상 결과가 나오는지 확인, 적합한 답변을 했거나 적합한 답변을 하지 않았다는 결과 도출
#선택한 결과 입력
#결과비교
#결과 도출

#def theme_anaysis(text_input):
#질문 7개
def qanda_analysis(input_text):
    
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
    
    quada_dic = {}
    
    for temp in sims : 
        
        quada_dic[temp[0]] = round(float(temp[1]),3)
        
        # result_sims.append([temp[0],round(float(temp[1]),3)])

    return quada_dic





#Connectives analysis
def connecitve_words_ratio(text):

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

    #OVERLAP 단어들을 리스트에 넣어서 필터로 만들고
    connecitve_words_list = ['after a while','afterwards','at once','at this moment'
                        ,'at this point','before that','finally','first'
                        ,'second','third','here','in the end','lastly'
                        ,'later on','meanwhile','next','next time','now'
                        ,'on another occasion','since','soon','straightaway','then'
                        ,'until then','when','whenever','while','besides'
                        ,'e.g.','for example','for instance','i.e.','in other words'
                        ,'in that','that is to say','first','second','firstly'
                        ,'secondly','first of all','finally','lastly','for one thing'
                        ,'for another','in the first place','to begin with','next','in summation','to conclude'
                        ,'additionally','also','as well','even','furthermore','in addition','indeed','let alone'
                        ,'moreover','not only','accordingly','all the same','an effect of','an outcome of','an upshot of'
                        ,'as a consequence of','as a result of','because','caused by','consequently'
                        ,'despite this','even though','hence','however','in that case','moreover'
                        ,'nevertheless','otherwise','so','so as','stemmed from','still'
                        ,'then','therefore','though','under the circumstances'
                        ,'yet','alternatively','anyway','but','by contrast','differs from'
                        ,'elsewhere','even so','however','in contrast','in fact'
                        ,'in other respects','in spite of this','in that respect','instead','nevertheless'
                        ,'on the contrary','on the other hand','rather','though','whereas'
                        ,'accordingly','as a result','as exemplified by','consequently','for example'
                        ,'for instance','for one thing','including','provided that','since'
                        ,'so','such as','then','therefore','these include','through','unless','without'
                     ]
    
    ####문장에 char_list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
    #우선 토큰화한다.
    retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
    token_input_text = retokenize.tokenize(essay_input_corpus)
    #print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
    #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
    filtered_chr_text = []
    for k in token_input_text:
        for j in connecitve_words_list:
            if k == j:
                filtered_chr_text.append(j)
    
    #print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
    
    filtered_chr_text_ = set(filtered_chr_text) #중복제거
    filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
    #print (filtered_chr_text__) # 중복값 제거 확인
    
    for i in filtered_chr_text__:
        ext_sim_words_key = model.wv.most_similar_cosmul(i) #모델적용(self.wv.most_similar_cosmul()로 수정완료)
    
    char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 캐릭터 표현 수
    char_count_ = len(filtered_chr_text__) #중복제거된 캐릭터 표현 총 수
        
    result_char_ratio = round(char_total_count/total_words * 100, 2)

    #return result_char_ratio, total_sentences, total_words, char_total_count, char_count_, ext_sim_words_key
    return result_char_ratio



# def pronouns_Anaphoric(text): >>> 전체 글에서 얼마나 포함되어 있는지 비율 도출
# def qanda_analysis(input_text): >>> 질문에 적절한 답을 제시해주는지 분석해준다.
# def connecitve_words_ratio(text): >>> 연결어에 대한 전체 문장당 포함 비율을 도출해준다.
# results = readability.getmeasures(text_input, lang='en') >>> 가독성 등 모든 것을 분석해준다.




def ai_theme (text_input) : 
    
    
    re_pronouns_Anaphoric = pronouns_Anaphoric(text_input)
    print("Anaphoric ratio :", re_pronouns_Anaphoric)

    ###########################################################################
    
    
    
    ##################################################################
    text = nltk.word_tokenize(text_input) #토크나이징
    pos_tagged = nltk.pos_tag(text) # 품사 태그 적용
    #명사 수 추출
    nouns = list(filter(lambda x:x[1]=='NN',pos_tagged))
    nouns_no = len(nouns)
    print ("nouns ration : ",nouns_no) #전체 문장에서 명사가 차지하는 비율 계산할 것
    #동사 수 추출
    vbzs = list(filter(lambda x:x[1]=='VBZ',pos_tagged))
    vbzs_no  = len(vbzs)
    print ("verb ration : ", vbzs_no) #전체 문장에서 동사가 차지하는 비율 게산할 것


    qanda_ = qanda_analysis(text_input)
    print("Q&A matching analysis :", qanda_)
    ##################################################################


    re_connecitve = connecitve_words_ratio(text_input)
    print ("Connective words ratio: ", re_connecitve)



    # re_pronouns_Anaphoric >>> 전체 글에서 얼마나 포함되어 있는지 비율 도출
    # qanda_                >>> 질문에 적절한 답을 제시해주는지 분석해준다.
    # re_connecitve         >>> 연결어에 대한 전체 문장당 포함 비율을 도출해준다.
    
    
    # readability grades:
    #     Kincaid:                          5.44
    #     ARI:                              6.39
    #     Coleman-Liau:                     6.91
    #     FleschReadingEase:               85.17
    #     GunningFogIndex:                  9.86
    #     LIX:                             31.98
    #     SMOGIndex:                        9.39
    #     RIX:                              2.56
    #     DaleChallIndex:                   8.02
    # sentence info:
    #     characters_per_word:              4.17
    #     syll_per_word:                    1.24
    #     words_per_sentence:              16.35
    #     sentences_per_paragraph:         11.5
    #     type_token_ratio:                 0.09
    #     characters:                  551335
    #     syllables:                   164205
    #     words:                       132211
    #     wordtypes:                    12071
    #     sentences:                     8087
    #     paragraphs:                     703
    #     long_words:                   20670
    #     complex_words:                10990
    #     complex_words_dc:             29908
    # word usage:
    #     tobeverb:                      3907
    #     auxverb:                       1630
    #     conjunction:                   4398
    #     pronoun:                      18092
    #     preposition:                  19290
    #     nominalization:                1167
    # sentence beginnings:
    #     pronoun:                       2578
    #     interrogative:                  217
    #     article:                        629
    #     subordination:                  120
    #     conjunction:                    236
    #     preposition:                    397

        

    #가독성 분석
    #https://pypi.org/project/readability/


    results = readability.getmeasures(text_input, lang='en')
    #readability grades:
    print("readability grades: FleschReadingEase = ",round(results['readability grades']['FleschReadingEase'],3))
    print("readability grades: Kincaid = ", round(results['readability grades']['Kincaid'],2))
    print("readability grades: ARI =", round(results['readability grades']['ARI'],2))
    print("readability grades: Coleman-Liau = ", round(results['readability grades']['Coleman-Liau'],2))
    print("readability grades: GunningFogIndex = ", round(results['readability grades']['GunningFogIndex'],2))
    print("readability grades: LIX = ", round(results['readability grades']['LIX'],2))
    print("readability grades: SMOGIndex = ", round(results['readability grades']['SMOGIndex'],2))
    print("readability grades: RIX = ", round(results['readability grades']['RIX'],2))
    print("readability grades: DaleChallIndex = ", round(results['readability grades']['DaleChallIndex'],2))
    #sentence info:
    print("sentence info: sentences = ", results['sentence info']['sentences'])
    print("sentence info: paragraphs = ", results['sentence info']['paragraphs'])
    print("sentence info: characters_per_word = ", round(results['sentence info']['characters_per_word'],2))
    print("sentence info: characters = ", results['sentence info']['characters'])
    print("sentence info: syllables = ", results['sentence info']['syllables'])
    print("sentence info: complex_words = ", results['sentence info']['complex_words'])
    #word usage:
    print("word usage : tobeverb =", results['word usage']['tobeverb'])
    print("word usage : auxverb =", results['word usage']['auxverb'])
    print("word usage : conjunction =", results['word usage']['conjunction'])
    print("word usage : pronoun =", results['word usage']['pronoun'])
    print("word usage : preposition =", results['word usage']['preposition'])
    print("word usage : nominalization =", results['word usage']['nominalization'])
    #sentence beginnings:
    print("sentence beginnings : pronoun =", results['sentence beginnings']['pronoun'])
    print("sentence beginnings : interrogative =", results['sentence beginnings']['interrogative'])
    print("sentence beginnings : article =", results['sentence beginnings']['article'])
    print("sentence beginnings : subordination =", results['sentence beginnings']['subordination'])
    print("sentence beginnings : conjunction =", results['sentence beginnings']['conjunction'])
    print("sentence beginnings : preposition =", results['sentence beginnings']['preposition'])
    
    # ai_theme = []
        
    # ai_theme = [
            
    #     ["re_pronouns_Anaphoric" , round(float(re_pronouns_Anaphoric),3)],
    #     ["qanda_" , qanda_],
    #     ["re_connecitve" , round(float(re_connecitve),3)],
        
    #     ["FleschReadingEase" , round(float(results['readability grades']['FleschReadingEase']),2)],
    #     ["Kincaid" , round(float(results['readability grades']['Kincaid']),2)],
    #     ["ARI" , round(float(results['readability grades']['ARI']),2)],
    #     ["Coleman_Liau" , round(float(results['readability grades']['Coleman-Liau']),2)],
    #     ["GunningFogIndex" , round(float(results['readability grades']['GunningFogIndex']),2)],
    #     ["LIX" , round(float(results['readability grades']['LIX']),2)],
    #     ["SMOGIndex" , round(float(results['readability grades']['SMOGIndex']),2)],
    #     ["RIX" , round(float(results['readability grades']['RIX']),2)],
    #     ["DaleChallIndex" , round(float(results['readability grades']['DaleChallIndex']),2)],
        
    #     ["sentences" , results['sentence info']['sentences']],
    #     ["paragraphs" , results['sentence info']['paragraphs']],
    #     ["characters_per_word" , round(float(results['sentence info']['characters_per_word']),2)],
    #     ["characters" , results['sentence info']['characters']],
    #     ["syllables" , results['sentence info']['syllables']],
    #     ["complex_words" , results['sentence info']['complex_words']],
    #     ["tobeverb" , results['word usage']['tobeverb']],
    #     ["auxverb" , results['word usage']['auxverb']],
    #     ["conjunctions" , results['word usage']['conjunction']],
    #     ["pronouns" , results['word usage']['pronoun']],
    #     ["prepositions" , results['word usage']['preposition']],
    #     ["nominalization" , results['word usage']['nominalization']],
    #     ["pronoun" , results['sentence beginnings']['pronoun']],
    #     ["interrogative" , results['sentence beginnings']['interrogative']],
    #     ["article" , results['sentence beginnings']['article']],
    #     ["subordination" , results['sentence beginnings']['subordination']],
    #     ["conjunction" , results['sentence beginnings']['conjunction']],
    #     ["preposition" , results['sentence beginnings']['preposition']]
        
    # ]
    
    
    
    ai_theme = {
            
        "re_pronouns_Anaphoric" : round(float(re_pronouns_Anaphoric),3),
        "qanda" : qanda_,
        "re_connecitve" : re_connecitve,
        "FleschReadingEase" : round(float(results['readability grades']['FleschReadingEase']),2),
        "Kincaid" : round(float(results['readability grades']['Kincaid']),2),
        "ARI" : round(float(results['readability grades']['ARI']),2),
        "Coleman_Liau" : round(float(results['readability grades']['Coleman-Liau']),2),
        "GunningFogIndex" : round(float(results['readability grades']['GunningFogIndex']),2),
        "LIX" : round(float(results['readability grades']['LIX']),2),
        "SMOGIndex" : round(float(results['readability grades']['SMOGIndex']),2),
        "RIX" : round(float(results['readability grades']['RIX']),2),
        "DaleChallIndex" : round(float(results['readability grades']['DaleChallIndex']),2),
        "sentences" : results['sentence info']['sentences'],
        "paragraphs" : results['sentence info']['paragraphs'],
        "characters_per_word" : round(float(results['sentence info']['characters_per_word']),2),
        "characters" : results['sentence info']['characters'],
        "syllables" : results['sentence info']['syllables'],
        "complex_words" : results['sentence info']['complex_words'],
        "tobeverb" : results['word usage']['tobeverb'],
        "auxverb" : results['word usage']['auxverb'],
        "conjunctions" : results['word usage']['conjunction'],
        "pronouns" : results['word usage']['pronoun'],
        "prepositions" : results['word usage']['preposition'],
        "nominalization" : results['word usage']['nominalization'],
        "pronoun" : results['sentence beginnings']['pronoun'],
        "interrogative" : results['sentence beginnings']['interrogative'],
        "article" : results['sentence beginnings']['article'],
        "subordination" : results['sentence beginnings']['subordination'],
        "conjunction" : results['sentence beginnings']['conjunction'],
        "preposition" : results['sentence beginnings']['preposition']
        
    }
    
    
    
    
    return ai_theme


text_input = """I inhale deeply and blow harder than I thought possible, pushing the tiny ember from its resting place on the candle out into the air. The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

print(ai_theme(text_input))

print("qanda_:",qanda_analysis(text_input))