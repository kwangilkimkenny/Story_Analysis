# Perspective 분석 - 나의 관점, 나의 의견을 표현하는 문구 분석
# 전체 문장에 위 표현이 얼마나 표현되어 있는지 계산하고, 합격생들의 전체 평균값과 비교하여 최종 결과를 수치로 계산

# Perspective (20%) - 나의 관점, 나의 의견 표출
# 로직: Opinion/Belief words (70%) + viewpoint adverbs (30%)  + emphasizing adjectives 

import re
import string
import nltk
#nltk.download()
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')



# preprocessing
def preprocessing(essay_input):
    essay_input_corpus = str(essay_input) #문장입력
    essay_input_corpus_ = essay_input_corpus.lower()#소문자 변환
    #print('essay_input_corpus :', essay_input_corpus)

    sentences  = sent_tokenize(essay_input_corpus_) #문장 토큰화 > 문장으로 구분
    total_sentences = len(sentences)#토큰으로 처리된 총 문장 수
    total_words = word_tokenize(essay_input_corpus_)
    total_words_num = len(word_tokenize(essay_input_corpus_))# 총 단어수
    #print(total_words)
    split_sentences = []
    for sentence in sentences:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences.append(words)

    # 총 문장수 곗한
    split_sentences_cnt = len(split_sentences)


    lemmatizer = WordNetLemmatizer()
    preprossed_sent_all = [] # 문장별로 단어의 원형을 리스트로 구분하여 변화한 모든 문장값 [[문장],[문장..토큰화] ...
    for i in split_sentences:
        preprossed_sent = [] # 개별문장을 단어 원형으로 구성된 리스트[문장..토큰화]
        for i_ in i:
            if i_ not in stop: #remove stopword
                lema_re = lemmatizer.lemmatize(i_, pos='v') #표제어 추출, 동사는 현재형으로 변환, 3인칭 단수는 1인칭으로 변환
                if len(lema_re) > 3: # 단어 길이가 3 초과단어만 저장(길이가 3 이하는 제거)
                    preprossed_sent.append(lema_re)
        preprossed_sent_all.append(preprossed_sent)

    #print('preprossed_sent_all:', preprossed_sent_all)
    
    # preprossed_sent_all 이중 리스트를 flatten하게 만들고, 여기에서 Perspective를 카운트해서 비교 계산하면 됨
    flatten_dic_list = [y for x in preprossed_sent_all for y in x]

        
    input_essay_words = flatten_dic_list
    #print('input_essay_words:', input_essay_words)


    # 문장들을 문장리스트로 분해하고 다시 각 문장을 단어 리스트로 분해하여 이중리스트를 만든다.
    def double_list(essay_input_data):
        result = essay_input_data.split(".")

        wdt_sents = []
        for i in result:
            wdt = word_tokenize(i)
            wdt_sent = []
            for j in wdt:
                lw_wdt = j.lower() # 소문자 변환
                wdt_sent.append(lw_wdt)
            wdt_sents.append(wdt_sent)
        return wdt_sents

    get_double_list = double_list(essay_input)

    # 결과 해석 #
    # input_essay_words : 입력한 에세이를 단어로 분해한 것 type: list ------------> 이것만 사용함
    # split_sentences_cnt : 입력한 총 문장의 수
    # get_double_list : 문장들을 문장리스트로 분해하고 다시 각 문장을 단어 리스트로 분해하여 이중리스트
    return input_essay_words, split_sentences_cnt, get_double_list



# 입력한 문장을 단어로 바꾼(words_list_input) 후, Perspective를 카운트해서 비교 계산하자
# wdt_sents_essay : 입력문장을 리스트로 변환한 값
def compare(words_list_input, essay_input):
    # Perspective words로 비교할 단어다
    perspective = ['seems','point','magnificent','confident','given','superior','notably','seen','subject','total','beautiful',
                    'repeatedly','come','either','position','sound','mean','light','think?','usually','delicious','rightly','reaction','wicked',
                    'expert','it’s','doubtless','evidence','go','surely','really','methinks','cannot','matter.','giving','sure','naturally',
                    'carelessly','fabulous','ask','said','estimation','quite','large','typically','complete','unexpected','never','suppose',
                    'may','care','reckon','unforeseeable','minority','complex','lovely','believe','superb','maybe','primarily','mind',
                    'observed','possibly','generally','probably','consideration','safely','picturesque','like','great','likely','imagine',
                    'consider','bet','standpoint','it’s','sheer','wisely','undoubtedly','prof','tell','opposite',"i'd",'suggests','i’d',
                    'initial','incredibly','simply','deadly','main','merely','amazingly','adverbs','staggering','idea','sensational','dare',
                    'eye','you’d','seriously','expressed','high','agree','well','positively','pleasant','question','perhaps','unpredictable',
                    'conviction','marvellous','excellent','unforeseen','old','thoughtfully','mainly','consistently','sufficiently','astonishing',
                    'truthfully','glorious','feeling','idiot','definitely','extraordinary','support','quality','topic','solely','understanding',
                    'satisfying','everyone','increasingly','pleasurable','obvious','might','resembles','doubt','delightful','phenomenal','must',
                    'conceit','ridiculously','appears','take','wa','find','unduly','opinion','honest','impression','confidentially','disagree',
                    'constantly','help','change',"i'm",'way','think','suspect','stunning','could','money','i’ll','complicated','course','alone',
                    'least','heavily','judgement','one','due','clear','tend','highly','positive','according','enjoyable','exclusively','bravely',
                    'argue','book','precisely','plainly','chiefly','i’ve','best','cleverly','difficult','top','predominantly','personally','wrote',
                    'can’t','far','taste','suggest','enormously','prime','unfortunately','unique','summarise','pretend','purely','want',
                    'reservation','deny','controversial','rather','right','pleasing','perspective','outta','dreadfully','kindly','indeed',
                    'presumably','exactly','wonderful',"one's",'frequently','know','regard','unbelievably','certain','pretty','fulfilling',
                    'obviously','matter','although','first-rate','sight','conceivably','sit','awesome','terrific','particularly','gather',
                    'thinking','postulate','familiar','perfectly','look','unbelievable','mistaken','vantage','grand','one','conclusion','part',
                    'imho','speaking','conclude','standing','classic','humble','scenic','won’t','stupidly','stand','frank','impressive',
                    'reasonable','spectacular','view','side','absolute','maintain','would','infer','serious','outstanding','mixed','fact',
                    'breathtaking','person','say','unlikely','neither','shred','personal','belief','sitting','know','weighing','fortunately',
                    'wish','sterling','without','absolutely','admit','utter','mostly','completely','attractive','majestic','totally','shadow',
                    'continually','sake','personally','people','towards','fair','claimed','technically','presume','truly','commenting','noticed',
                    "i'm",'exactly.','viewpoint','crazy','speaking','assume','always','limited','much','theoretically','head','gratifying',
                    'fantastic','incredible','read','get','reckoning','bitterly','even','correct','without','remarkable','knowledge','tremendous',
                    'obviously','exquisite','disappointingly','thought','assumes','especially','miraculous','rewarding','strongly','convinced',
                    'concerned','exceptional','case','argument','actual','amazing','i’m','surprising','imposing','charming','entirely','hold',
                    'quite','pure','clearly','brilliant','sat','fairly','assessment','understand','surprisingly','seem','guess','generously',
                    'see','certainly','foolishly','whole','particular','perfect','wrong','issue','extremely','judgment','argued','frankly',
                    'luckily','experience','feel','this','frankly','saying','later','situation','unusual']

    viewpoint_wd = ['carelessly','bravely','definitely','clearly','circus','theoretically','silly','definitely','confidentially','stupidly','disappointingly','sally',
                    'presumably','bravely','kindly','generously','definitely','silly','naturally','clearly','undoubtedly','obviously','kindly','theoretically',
                    'personally','certainly','stupidly','unfortunately','personally','truthfully','thoughtfully','defintely','simply','seriously','foolishly','theoretically',
                    'fortunately','obviously','cleverly','thoughtfully','surely','stupidly','sally','technically','confidentially','technically','fortunately','luckily',
                    'undoubtedly','foolishly','surely','naturally','wisely','personally','naturally','wisely','thoughtfully',
                    'carelessly','unfortunately','undoubtedly','presumably','clearly','generously','unbelievably','cleverly','obviously','carelessly','generously','wisely',
                    'unbelievably','seriously','surely','technically','kindly','foolishly','presumably','truthfully','luckily','unfortunately','unbelievably','disappointingly',
                    'seriously','disappointingly','bravely','confidentially','cleverly','truthfully','luckily','certainly','fortunately','certainly','simply']

    emphasis_wd = ['absolute', 'complete', 'outright', 'pure', 'real', 'total', 'true', 'utter']

    sentences_capital = ['In my opinion','I believe','In my mind','It would seem that ','It could be argued that','The evidence suggests that',
                    'This proves that the','supports the idea that','Although ','It seems to me that','In my opinion,','I am of the opinion that','I take the view that',
                    'My personal view is that','In my experience','As far as I understand','As far as I can see','As I see it','From my point of view','As far as I know',
                    'From what I know','I might be wrong but','If I am not mistaken','I believe one can say','I believe one can safely say','I cannot deny that',
                    'I can imagine that','I think/believe/suppose','Personally, I think','That is why I think','I am sure that','I am certain that','I am convinced that',
                    'I am not sure, but', 'I am not certain, but',
                    "I am not sure, because I don't know the situation exactly",'I am not convinced that','I have read that','I am of mixed opinions about',
                    'I am of mixed opinions on','It is obvious that','It is certain that','One can say that','It is clear that','There is no doubt that','The fact is that',
                    'The point is that','The main point is that','This proves that','What it comes down to is that','I am of mixed opinions on this',
                    'I am of mixed opinions about this','I have no opinion in this matter','It is claimed that','I must admit that','in my opinion','from my point of view',
                    'in my view','as i see','i think','my mind','as i see it','from my standpoint','i believe','i believe that','from my vantage point','from where i sit',
                    'from where i stand','if you ask me','way i see it','as far as i am concerned','for me','in my eyes','in my point of view','i think that','in my estimation',
                    'in my perspective','it seems to me','i feel','in my judgment','according to my way of thinking','from my own point of view','from my viewpoint','in my book',
                    'in my judgement','way i see','according to me','according to my belief','from a personal perspective','from my view','i suppose','it seems to me that',
                    'speaking for myself','speaking personally','for my part','for myself','i feel that','my way of thinking',"as far as i'm concerned",'for my money',
                    'i assume','my thinking','my viewpoint','according to my lights','as for me','by my reckoning','for my own part','from me',"from one's point of view",
                    'from our point of view',"from where i'm sitting",'how i see it','i am of the opinion that','in thinking this through, i conclude','outta my sight',
                    'i consider','i do believe','i get the feeling that','i get the impression that','i guess','i have a feeling that','i have the feeling that',
                    'i have the impression that','i see','i suspect','i thought','i was thinking','in my experience','in my mind','in my own view','it is my belief that',
                    'it seems that','my eyes','my way','the best of my belief','as far as i can tell','as far as i know','as far as i understand','as far as i was concerned',
                    'as i wish','as i would have it','consider this','for my taste','from my personal standpoint','from the perspective','from where i was standing',
                    "from where i'm standing",'i believe so','i guess that','i have noticed that','i have observed','i have observed that','i have seen that','i maintain that',
                    'i reckon','i regard','i suggest','i would say','i would suggest','in my humble opinion','in my own conceit','in my position','in my thinking','it seems likely',
                    "it's my understanding",'methinks','my understanding is','my view is that','sounds to me like','what i say is','you can see that','as far as i understand it',
                    'as to me','by my estimation','by my lights','for all i know','for my own sake','for my sake','from where i am','i conclude','i dare say that','i imagine that',
                    'i infer','i personally believe','i personally feel','i personally find','i personally suppose','i personally think','i presume','i reckon that','i understand that',
                    'i would say that',"i'd like to point out that","i'd suggest that","i'm thinking",'in my own viewpoint','in my personal opinion','in opinion','it appears',
                    'it goes without saying that','it is my assessment','it is my impression that','it is my opinion that','it looks like','it pretends to be','it resembles',
                    'it seems',"it's obvious to me",'my belief is that','my conviction is that','my impression is that','my knowledge','my opinion is that','my perspective',
                    'my thinking is','my thought is','my view','one can postulate','one might argue that','personally','personally i think','personally speaking','point is that',
                    'this assumes','what i mean is','all i care','for all i care','from my side','from my sight','from sight','from where i sat','from where i was sitting','in my case',
                    'in my personal opinion, i believe','in my personal opinion, i reckon','in my personal opinion, i suppose','in my personal opinion, i think','in my viewpoint',
                    'I think that','I really think that','I believe','I believe that','I’m sure that','In my opinion','My opinion is','I agree with','I feel that','I imagine','I guess',
                    'I have no doubt that','I’m certain that','I strongly believe that','I’ve never really thought about this before, but','My personal opinion is that',
                    'Personally, my opinion is that ','To be honest','In my honest opinion, ','As far as I know, ','I agree with the opinion of ','I could be wrong, but ',
                    'I’d definitely say that ','I’d guess that ','I’d imagine that ','I’d say that','I’m absolutely certain that','I’m fairly confident that','I’m no expert, but',
                    'I’m no expert on this, but ','I’m positive that','I’m pretty sure that','It seems to me that','It’s a complicated/difficult issue, but','My view is',
                    'My view on this is','My point of view on this is','My point of view is','Obviously,','To the best of my knowledge,','What I think is','You could say',
                    'My opinion was best expressed by ','In my limited experience','It could/might well be that ','Know what I think? ','In my opinion','In my eyes','To my mind',
                    'As far as I am concerned','Speaking personally','From my point of view','As for me','As to me','My view is that','My opinion is that','My beliefis that',
                    'My impression is that','My conviction is that','I hold the view that','I am sure','I am certain that ','Some people may disagree with me, but',
                    'This is just my opinion, but','Without a doubt,','You probably won’t agree, but','After much thought,','After weighing up both sides of the argument',
                    'Although I can see both points of view','Although I can understand the opposite point of view','As I see it,','Correct me if I’m wrong, but','For me',
                    'From my point of view','Frankly,','I am not very familiar with this topic, but','I do believe','I do feel','I do think','I have come to the conclusion that',
                    'I might change my mind later, but','I suppose','I reckon','I tend to think thaㅅ','I’m not sure I’m the right person to ask, but',
                    'I have very limited experience of this, but','I’m pretty confident that','I’ve always thought that','If you ask me,',"I'm convinced that","I'm absolutely convinced that",
                    'In my humble opinion','IMHO','It could be said that','It seems clear to me that','It would seem to me that','My initial reaction is',
                    'Not everyone will/would agree with me, but','Personally speaking','Speaking for myself','I would say that','It seems to me that','I am of the opinion that',
                    'My impression is that','I am under the impression that','It is my impression that','I have the feeling that   ','My own feeling on the subject is that',
                    'I have no doubt that','My view this issue is clear','My position on this issue is that','My view on this is that','Off the top of my head','Plainly',
                    'Quite frankly','There is a part of me that says','This may well be controversial, but','To my mind','To my way of thinking','To summarise my views on the matter',
                    'To summarise my rather complex views on the matter','What I always say is','With some reservations','Without a shred of doubt','Without a shadow of doubt',
                    'You’d have to be crazy not to agree thatAny idiot can see that','After giving this matter some (serious) thought',"As far as I'm concerned",'As the old saying goes',
                    'Having given this question due consideration','I am of the opinion that','I can’t help thinking that','I know this is a minority view, but ',
                    'I’m in the minority in thinking that','I tend towards the opinion that','I think it’s reasonable to say','I think it’s fair to say','I’ll tell you what I think',
                    'I’m quite convinced that','I’m entirely convinced that','I’ve come the conclusion that','If I must come up with an opinion','If you want my opinion,',
                    'The way I see it is','If you really want my opinion,','The way I see it','To be frank','To be perfectly frank','I think that','I consider that','I find that',
                    'I feel that','I believe  that','I suppose that','I presume  that','I assume that','I hold the opinion that','I dare say that','I guess that','I bet that',
                    'I gather that','It goes without saying that']

    sentence_lower = ['in my opinion','i believe','in my mind','it would seem that','it could be argued that','the evidence suggests that','this proves that the','supports the idea that',
                    'although ','it seems to me that','in my opinion,','i am of the opinion that','i take the view that','my personal view is that','in my experience','as far as i understand',
                    'as far as i can see','as i see it','from my point of view','as far as i know','from what i know','i might be wrong but','if i am not mistaken','i believe one can say',
                    'i believe one can safely say','i cannot deny that','i can imagine that','i believe','i think', 'i suppose','personally, i think','that is why i think','i am sure that',
                    'i am certain that','i am convinced that','i am convinced that','i am not sure, but','i am not certain, but',
                    "i am not sure, because i don't know the situation exactly",'i am not convinced that','i have read that','i am of mixed opinions about',
                    'i am of mixed opinions on','it is obvious that','it is certain that','one can say that','it is clear that','there is no doubt that','the fact is that','the point is that',
                    'the main point is that','this proves that','what it comes down to is that','i am of mixed opinions on this','i am of mixed opinions about this','i have no opinion in this matter',
                    'it is claimed that','i must admit that','in my opinion','from my point of view','in my view','as i see','i think','my mind','as i see it','from my standpoint','i believe',
                    'i believe that','from my vantage point','from where i sit','from where i stand','if you ask me','way i see it','as far as i am concerned','for me','in my eyes','in my point of view',
                    'i think that','in my estimation','in my perspective','it seems to me','i feel','in my judgment','according to my way of thinking','from my own point of view','from my viewpoint',
                    'in my book','in my judgement','way i see','according to me','according to my belief','from a personal perspective','from my view','i suppose','it seems to me that',
                    'speaking for myself','speaking personally','for my part','for myself','i feel that','my way of thinking',"as far as i'm concerned",'for my money','i assume','my thinking',
                    'my viewpoint','according to my lights','as for me','by my reckoning','for my own part','from me',"from one's point of view",'from our point of view',"from where i'm sitting",
                    'how i see it','i am of the opinion that','in thinking this through, i conclude','outta my sight','i consider','i do believe','i get the feeling that','i get the impression that',
                    'i guess','i have a feeling that','i have the feeling that','i have the impression that','i see','i suspect','i thought','i was thinking','in my experience','in my mind',
                    'in my own view','it is my belief that','it seems that','my eyes','my way','the best of my belief','as far as i can tell','as far as i know','as far as i understand',
                    'as far as i was concerned','as i wish','as i would have it','consider this','for my taste','from my personal standpoint','from the perspective','from where i was standing',
                    "from where i'm standing",'i believe so','i guess that','i have noticed that','i have observed','i have observed that','i have seen that','i maintain that','i reckon',
                    'i regard','i suggest','i would say','i would suggest','in my humble opinion','in my own conceit','in my position','in my thinking','it seems likely',"it's my understanding",
                    'methinks','my understanding is','my view is that','sounds to me like','what i say is','you can see that','as far as i understand it','as to me','by my estimation',
                    'by my lights','for all i know','for my own sake','for my sake','from where i am','i conclude','i dare say that','i imagine that','i infer','i personally believe',
                    'i personally feel','i personally find','i personally suppose','i personally think','i presume','i reckon that','i understand that','i would say that',"i'd like to point out that",
                    "i'd suggest that","i'm thinking",'in my own viewpoint','in my personal opinion','in opinion','it appears','it goes without saying that','it is my assessment','it is my impression that',
                    'it is my opinion that','it looks like','it pretends to be','it resembles','it seems',"it's obvious to me",'my belief is that','my conviction is that','my impression is that',
                    'my knowledge','my opinion is that','my perspective','my thinking is','my thought is','my view','one can postulate','one might argue that','personally','personally i think',
                    'personally speaking','point is that','this assumes','what i mean is','all i care','for all i care','from my side','from my sight','from sight','from where i sat',
                    'from where i was sitting','in my case','in my personal opinion, i believe','in my personal opinion, i reckon','in my personal opinion, i suppose','in my personal opinion, i think',
                    'in my viewpoint','i think that','i really think that','i believe','i believe that','i’m sure that','in my opinion','my opinion is','i agree with','i feel that','i imagine',
                    'i guess','i have no doubt that','i’m certain that','i strongly believe that','i’ve never really thought about this before, but','my personal opinion is that',
                    'personally, my opinion is that ','to be honest','in my honest opinion, ','as far as i know, ','i agree with the opinion of ','i could be wrong, but ','i’d definitely say that',
                    'i’d guess that ','i’d imagine that ','i’d say that','i’m absolutely certain that','i’m fairly confident that','i’m no expert, but','i’m no expert on this, but ','i’m positive that',
                    'i’m pretty sure that','it seems to me that','it’s a complicated issue, but','it’s a difficult issue, but','my view is','my view on this is','my point of view on this is','my point of view is',
                    'obviously,','to the best of my knowledge,','what i think is','you could say','my opinion was best expressed by ','in my limited experience','it could/might well be that',
                    'know what i think? ','in my opinion','in my eyes','to my mind','as far as i am concerned','speaking personally','from my point of view','as for me','as to me','my view is that',
                    'my opinion is that','my beliefis that','my impression is that','my conviction is that','i hold the view that','i am sure','i am certain that ','some people may disagree with me, but',
                    'this is just my opinion, but','without a doubt,','you probably won’t agree, but','after much thought,','after weighing up both sides of the argument',
                    'although i can see both points of view','although i can understand the opposite point of view','as i see it,','correct me if i’m wrong, but','for me','from my point of view',
                    'frankly,','i am not very familiar with this topic, but','i do believe','i do feel','i do think','i have come to the conclusion that','i might change my mind later, but',
                    'i suppose','i reckon','i tend to think that','i’m not sure i’m the right person to ask, but','i have very limited experience of this, but','i’m pretty confident that',
                    'i’ve always thought that','if you ask me,',"i'm convinced that","i'm absolutely convinced that",'in my humble opinion','imho','it could be said that','it seems clear to me that',
                    'it would seem to me that','my initial reaction is','not everyone will agree with me, but','not everyone would agree with me, but','personally speaking','speaking for myself','i would say that',
                    'it seems to me that','i am of the opinion that','my impression is that','i am under the impression that','it is my impression that','i have the feeling that',
                    'my own feeling on the subject is that','i have no doubt that','my view this issue is clear','my position on this issue is that','my view on this is that','off the top of my head',
                    'plainly','quite frankly','there is a part of me that says','this may well be controversial, but','to my mind','to my way of thinking','to summarise my views on the matter',
                    'to summarise my rather complex views on the matter','what i always say is','with some reservations','without a shred/shadow of doubt',
                    'you’d have to be crazy not to agree thatany idiot can see that','after giving this matter some serious thought',"as far as i'm concerned",'as the old saying goes',
                    'having given this question due consideration','i am of the opinion that','i can’t help thinking that','i know this is a minority view, but ','i’m in the minority in thinking that',
                    'i tend towards the opinion that','i think it’s reasonable to say','i think it’s fair to say','i’ll tell you what i think','i’m quite convinced that',
                    'i’m entirely convinced that','i’ve come the conclusion that','if i must come up with an opinion','if you want my opinion,','the way i see it is',
                    'if you really want my opinion,','the way i see it','to be frank','to be perfectly frank','i think that','i consider that','i find  that','i feel that','i believe that',
                    'i suppose that','i presume that','i assume that','i hold the opinion that','i dare say that','i guess that','i bet that','i gather that','it goes without saying that']


    # 문장들을 문장리스트로 분해하고 다시 각 문장을 단어 리스트로 분해하여 이중리스트를 만드는 Method
    def sents_to_sent_list(sentence_lower):
        wdt_sents = []
        for i in sentence_lower:
            wdt = word_tokenize(i)
            wdt_sent = []
            for j in wdt:
                lw_wdt = j.lower() # 소문자 변환
                wdt_sent.append(lw_wdt)
            wdt_sents.append(wdt_sent)

        return wdt_sents

    # stop words를 제거한다. 그래야 결과가 정교해짐
    def proprocess_2nd(input_split_sentences):
        lemmatizer = WordNetLemmatizer()
        preprossed_sent_all = [] # 문장별로 단어의 원형을 리스트로 구분하여 변화한 모든 문장값 [[문장],[문장..토큰화] ...
        for i in input_split_sentences:
            preprossed_sent = [] # 개별문장을 단어 원형으로 구성된 리스트[문장..토큰화]
            for i_ in i:
                if i_ not in stop: #remove stopword
                    lema_re = lemmatizer.lemmatize(i_, pos='v') #표제어 추출, 동사는 현재형으로 변환, 3인칭 단수는 1인칭으로 변환
                    if len(lema_re) > 3: # 단어 길이가 3 초과단어만 저장(길이가 3 이하는 제거)
                        preprossed_sent.append(lema_re)
            preprossed_sent_all.append(preprossed_sent)
        return preprossed_sent_all


    comp_sents = sents_to_sent_list(sentence_lower)  # [['in', 'my', 'opinion'], ['i', 'believe'],['in', 'my', 'mind'],....
    comp_sents_ = proprocess_2nd(comp_sents)

    wwd = preprocessing(essay_input) # wwd[2] 의 결과를 아래에 넣음

    # 비교하려는 표현이 들어있는 문장을 찾기 위해서
    # 각 표현의 문장과 에세이 문장을 비교하여 겹치는 단어 수가 정확히 표현문장과 일치할 때, 원본 문장을 생성한다.
    get_matching_sents_all = [] #표현이 포함된 문장 추출되어 리스트에 담김
    for sent in comp_sents_: # 비교하고자 하는 문구를 리스트로 분리한 리스트집합
        for e in sent: # 문구별로 하나씩 가져와서, 개발 리스트의 값(단어)
            for itm in wwd[2]: # 에세이의 한 문장을 리스트로 분리한 값
                cnt = 0 #카운터 초기화
                get_matching_sents = []
                if e in itm: # 에세이 문장에, 비교 문구의 단어가 있다면, 카운트를 해서
                    cnt += 1
                    if cnt == len(sent)-2: #일치하는 단어가 비교하려는 문구의 단어 수와 같다면, 해당 비교 문장(단어 리스트로 묶인 문장)을 별도로 저장한다.
                        get_matching_sents.append(itm)
                    else:
                        pass
                get_matching_sents_all.append(get_matching_sents)

    # 빈 리스트 없애기 - 이중리스트 삭제하고
    get_mat_org_sents = [y for x in get_matching_sents_all for y in x] 

    # 각 리스트를 개별 문장으로 복원한 리스트로 변환
    full_snt_re = []
    for lit in get_mat_org_sents:
        full_snt = ' '.join(lit)
        full_snt_re.append(full_snt)
    # 추출한 최종 문장 - 이것은 비교분석하고자하는 문장이 포함된 문장이다.
    result_re = list(set(full_snt_re))
    #print('추출한 최종 문장 - 이것은 비교분석하고자하는 문장이 포함된 문장:', result_re)
    # perstive의 관련 단어가 포함된 문장의 수
    tot_cnt_perspective_exp = len(result_re)
    #print('perstive의 관련 단어가 포함된 문장의 수:', tot_cnt_perspective_exp)

    get_sentences_of_perspective_exp = [] # perspectiv 표현이 적용된 문장 '생성'! 여기에 사용된 문장을 단어로 구분하여 원본문장을 찾아서 웹이 표시하자
    for line in result_re:
        a = line[0].capitalize() # capitalize the first word of sentence  - 문장의 첫 단어를 대문자로 변환
        for i in range(1, len(line)):
            a = a + line[i] 
        #print('첫 단어를 대문자로 변환한 문장 리스트: ', a)
        get_sentences_of_perspective_exp.append(a)

    # 생성된 문장으로 원본 문장을 다시 찾아내는 코드 get_sentences_of_perspective_exp  -> 원본문장을 추출하자
    # 그러기 위해서는 이중리스트로(문장별로 단어 리스트로 처리해야하니깨) 변환해야함
    get_fin_sentences = sents_to_sent_list(get_sentences_of_perspective_exp) # 문장별로 리스트로 만듬. 즉, 2중리스트
    #print('get_fin_sentences:', get_fin_sentences) # 원본문장에서 get_fin_sentences 의 리스트 값을 비교하여 최소 단어가 4개이하로 일치하면 원본문장을 추출하여 저장하고, 최종 출력한다. 이것은 웹에 표시할거임

    #### perspective 표현이 들어간 단어가 포함된 원본문장 찾기 ####
    get_origin_sentences = [] 

    #print('essay_input:', essay_input)
    lines =  essay_input.split(".")
    #print('lines:', lines)
    for each_line in lines: # 원본문장에서 한 문장씩 꺼내서 반복해서 다 꺼내보자
        for each_sentence in get_fin_sentences: # 단어의 리스토로 변환된 문장 하나를 가져와서 각 단어를 꺼내서 포함 여부를 비교한다. 
            find_wd_cnt = 0 # 카운터 초기화
            for each_wd in each_sentence:
                #print('each_wd:', each_wd)
                if each_line.find(each_wd) > 0: # 하나 이상 발견되면
                    #print('each_line:', each_line)
                    find_wd_cnt += 1
                    #print('find_wd_cnt:', find_wd_cnt)
                    if find_wd_cnt >= 7: # 문장에 7단어 이상이 겹치면 같은 문장으로 판단
                        #print('each_line:', each_line)
                        get_origin_sentences.append(each_line) # 해당 원본 문장을 저장한다.
                    else:
                        pass
                else:
                    pass

    # 추출한 결과물 중복제거
    get_origin_unique_sentences = list(set(get_origin_sentences))


    # perspective 문장들(대문자, 소문자포함)  --- 이 값도 같은 부분 찾아내서 출력할 것  find 를 이용해서 일치하는 문자열 찾아내기
    perspective_sentences = sentences_capital + sentence_lower + perspective + viewpoint_wd + emphasis_wd # 찾고자하는 문자열 준비

    get_ps_sent_for_web = [] # perspective_sentences 의 표현 기반 웹에 표현할 perspective 단어 수집
    for perspective_each_expression in perspective_sentences:
        if perspective_each_expression in essay_input:
            get_ps_sent_for_web.append(perspective_each_expression)
        else:
            pass

    print('get_ps_sent_for_web:', get_ps_sent_for_web)
    # 최종 비교분석 결과  -- 웹에 표시함
    get_ps_sent_for_web_unique_re = list(set(get_ps_sent_for_web))





    input_essay_wds = preprocessing(essay_input)

    # 입력한 에세이의 총 문장 수 
    tot_sent_num = input_essay_wds[1]
    #print('입력한 에세이의 총 문장 수: ', tot_sent_num)


    # 전체 문장에서 perspective 관련 단어 사용 비율 계산
    sentence_usage_ratio = round((tot_cnt_perspective_exp / tot_sent_num) * 100, 2)

    # 결과 #
    # sentence_usage_ratio : pespective 관련 표현 사용 비율 ???????????? 불용어를 제거해보자.. 쓸데없은 표현은 없애보면 좀 더 결과가 정교해짐

    # get_origin_unique_sentences : perspective 표현이 사용된 원본문장 ----> 웹사이트에 표시해야 함
    # get_ps_sent_for_web_unique_re : perspective 표현을 준비한 lexicon을 이용해서 추출한 표현들 ---> lexicon을 이용하여 관련 표현이 있는 부분 추출, 이것도 웹사이트에 표시해야 함
    return sentence_usage_ratio, get_origin_unique_sentences, get_ps_sent_for_web_unique_re



### 실행 함수 ###
def getPerspectiveWD(essay_input):
    prepro_re = preprocessing(essay_input) # 전처리
    comp_re = compare(prepro_re[0],essay_input) # 계산
    return comp_re


# input College Supp Essay 
essay_input = """I inhale deeply and blow harder than I thought possible, pushing the tiny ember from its resting place on the candle out into the air. The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

## run ##

result = getPerspectiveWD(essay_input)
print('result:',result)


# return 값 예 #
# (24.0,  
# [" At home, if I mentioned that I had tried eggs for breakfast at a friend's house, ...  원본 문장이 주욱 추출됨"],  ----> 웹사이트에 표시해야 함
# ['In my mind', 'my mind', 'my perspective'])  ---> lexicon을 이용하여 관련 표현이 있는 부분 추출, 이것도 웹사이트에 표시해야 함