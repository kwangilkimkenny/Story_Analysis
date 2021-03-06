## 캐릭터 분석 단독 페이지를 구현한 코드 ##

# MBTI를 제외한 4개의 분석이 가능함
# 1000명의 데이터를 적용해서 평균값을 비교하지는 않았음 -- 해야지...


## 결과임  ##

# ===============================================================
# Character Descriptiveness :  33.0
# ===============================================================
# ['their', 'they', 'me', 'you', 'her', 'it', 'myself', 'i', 'my', 'them']
# ai_character_section.py:92: DeprecationWarning: Call to deprecated `most_similar_cosmul` (Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead).
#   ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
# =============================================
# Number of Characters : 92
# =============================================
# ['i', 'my', 'me']
# ai_character_section.py:338: DeprecationWarning: Call to deprecated `most_similar_cosmul` (Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead).
#   ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
# =============================================
# Emphasis on You : 60
# =============================================
# ['their', 'they', 'you', 'her', 'it', 'myself', 'them']
# ai_character_section.py:409: DeprecationWarning: Call to deprecated `most_similar_cosmul` (Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead).
#   ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
# =============================================
# Emphasis on Others : 32
# =============================================
# (33.0, 92, 60, 32)



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
        print (filtered_chr_text__) # 중복값 제거 확인
        
        for i in filtered_chr_text__:
            ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
        char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 캐릭터 표현 수
        char_count_ = len(filtered_chr_text__) #중복제거된 캐릭터 표현 총 수
            
        result_char_ratio = round(char_total_count/total_words * 100, 2)
        return char_total_count



    # number_of_characters = NumberofCharacters(input_text) # 문장에서 키워드와 관련된 단어을 모두 추출하면 이런 결과가 나옴, 이 결과를 모두 합쳐서 캐릭터 총 값 계산해서 숫자로 출력
    # number_of_characters
    # print ('=============================================')
    # print ('Number of Characters :', number_of_characters)
    # print ('=============================================')

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
                        'twin','uncle','widow','widower','wife','ex-wife']

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

            #print(test)
            dataf = pd.DataFrame(test, columns=['label', 'text'])
            #print(dataf)
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
            #     print("text:",text)
            #     print("label:",label)


            #저장된 모델을 불러온다.
            #J:\Django\EssayFit_Django\essayfitaiproject\essayfitapp\model.pt
            #time.sleep(1)
            #model = torch.load("/Users/jongphilkim/Desktop/Django_WEB/essayfitaiproject/essayai/model.pt", map_location=torch.device('cpu'))
            model = torch.load("model.pt", map_location=torch.device('cpu'))
            print("model loadling~")
            model.eval()


            pred_loader = test_loader
            print("pred_loader:", pred_loader)
            total_loss = 0
            total_len = 0
            total_showing__ = 0
            total_telling__ = 0

            showing_conunter = [] #문장에 해당하는 SHOWING을 계산한다.
            
            print("check!")
            for text, label in pred_loader:
                print("text:",text)
                #print("label:",label)
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
                print('pred :', pred)
                # correct = pred.eq(labels) 
                showing__ = pred.eq(1) # 예측한 결과가 1과 같으면 showing이다   >> TRUE   SHOWING을 추출하려면 이것만 카운드하면 된다. 
                telling__ = pred.eq(0) # 예측한 결과가 0과 같으면 telling이다   >> FALSE
                
                #print('showing : ', showing__)
                #print('telling : ', telling__)
                
                
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

        # print("===============================================================")
        # print ('Character Descriptiveness : ', showing_sentence_with_char)
        # print("===============================================================")

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
        character_list = ['i', 'I', 'my', 'me', 'mine']
        
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
        print (filtered_chr_text__) # 중복값 제거 확인
        
        for i in filtered_chr_text__:
            ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
        char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 캐릭터 표현 수
        char_count_ = len(filtered_chr_text__) #중복제거된 캐릭터 표현 총 수
            
        result_char_ratio = round(char_total_count/total_words * 100, 2)
        return char_total_count


    # EmphasisOnYou_ = EmphasisOnYou(input_text) # 문장에서 키워드와 관련된 단어을 모두 추출하면 이런 결과가 나옴, 이 결과를 모두 합쳐서 캐릭터 총 값 계산해서 숫자로 출력
    # EmphasisOnYou_
    # print ('=============================================')
    # print ('Emphasis on You :', EmphasisOnYou_)
    # print ('=============================================')



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
        character_list = ['you', 'your', 'they','them',
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
        print (filtered_chr_text__) # 중복값 제거 확인
        
        for i in filtered_chr_text__:
            ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
        char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 캐릭터 표현 수
        char_count_ = len(filtered_chr_text__) #중복제거된 캐릭터 표현 총 수
            
        result_char_ratio = round(char_total_count/total_words * 100, 2)
    
    
        return char_total_count


    # EmphasisOnOthers_ = EmphasisOnOthers(input_text) # 문장에서 키워드와 관련된 단어을 모두 추출하면 이런 결과가 나옴, 이 결과를 모두 합쳐서 캐릭터 총 값 계산해서 숫자로 출력
    # EmphasisOnOthers_
    # print ('=============================================')
    # print ('Emphasis on Others :', EmphasisOnOthers_)
    # print ('=============================================')


    character_descriptiveness = character_descrip(text)
    print("===============================================================")
    print ('Character Descriptiveness : ' , character_descriptiveness)
    print("===============================================================")


    number_of_characters = NumberofCharacters(text) 
    print ('=============================================')
    print ('Number of Characters :' , number_of_characters)
    print ('=============================================')


    EmphasisOnYou_ = EmphasisOnYou(text)
    print ('=============================================')
    print ('Emphasis on You :' , EmphasisOnYou_)
    print ('=============================================')


    EmphasisOnOthers_ = EmphasisOnOthers(text) 
    print ('=============================================')
    print ('Emphasis on Others :' , EmphasisOnOthers_)
    print ('=============================================')


    return character_descriptiveness, number_of_characters, EmphasisOnYou_, EmphasisOnOthers_




###################################################################
###################################################################



########## 실행 테스트 ##########

# 한명의 에세이 데이터 입력하여 계산
text= """A window into the soul.For most people, this would be the eyes. The eyes cannot lie; they often tell more about a person's emotions than their words. What distinguishes a fake smile from a genuine one? The eyes. What shows sadness? The eyes. What gives away a liar? The eyes.But are the eyes the only window into the soul?Recently, I began painting with watercolors. With watercolors, there is no turning back: if one section is too dark, it is nearly impossible to lighten the area again. Every stroke must be done purposefully, every color mixed to its exact value.I laid my materials before me, preparing myself for the worst. I checked my list of supplies, making sure my setup was perfect.I wet my brush, dipped it into some yellow ochre, and dabbed off the excess paint. Too little water on my brush. I dipped my brush back into my trusty water jar; the colors swirled beautifully, forming an abstract art piece before my eyes. \u2014It's a shame that I couldn't appreciate it.I continued mixing colors to their exact value. More alizarin crimson. More water. More yellow ochre. Less water. More phthalo blue. The cycle continued. Eventually, I was satisfied. The colors looked good, there was enough contrast between facial features, and the watercolors stayed inside the lines.Craving feedback, I posted my art to Snapchat. I got a few messages such as 'wow' and 'pretty,' but one message stood out. 'You were anxious with this one, huh? Anyways, love the hair!'I was caught off guard. Was it a lucky guess? Did they know something I didn't? I immediately responded: 'Haha, how could you tell?' No response.What I didn't know at the time was that my response would come a few months later while babysitting. Since the girl I was babysitting loved art, I took out some Crayola watercolors and some watercolor paper for her to play with. After I went to the bathroom and came back, the watercolors were doused with water. 'You were impatient with this one, huh? Anyways, love the little dog you drew!'The little girl looked up at me, confused. 'How could you tell?' 'You used a lot of water for a brighter color, but you couldn't wait for it to slowly soak in.''Oh.'Now, I would be lying if I said I realized the connection between the two events immediately.Instead, I made the connection when I decided to sit down one day and objectively critique my art. The piece that I once loved now seemed like a nervous wreck: the paper was overworked, the brushstrokes were undecided, the facial features blended together, and each drop of water was bound inside the lines as if it was a prisoner in a cage.From then on, I started noticing pieces of personality in additional creations surrounding me: website designs, solutions to math problems, code written for class, and even the preparation of a meal.When I peer around at people's projects during Code Club, I notice the clear differences between their code. Some people break it up by commenting in every possible section. Others breeze through the project, not caring to comment or organize their code. I could also see clear differences in personalities when our club members began coding the Arduino for the first time. Some followed the tutorials to the letter, while others immediately started experimenting with different colored LEDs and ways of wiring the circuit.It became clear to me that, as humans, we leave pieces of our souls in everything we do, more than we intend to. If we entertain this thought, perhaps the key to better understanding others around us is simply noticing the subtler clues under our noses?Perhaps there are endless windows to the soul, and we simply need to peer through them. I shakily rose my hand. 'We should create workshops of our own,' I suggested.I got a few strange looks. 'It's a good idea, but it's too much work.' 'We just don't have enough free time to make it work.' 'Maybe we could, but I don't know how to make workshops.' My suggestion was shot down. I shuffled in my seat. 'I could make them.' A few people stared at me in disbelief. I glanced over at the club advisor, Mr. C, nervous to hear his response.'If you're willing to take on the work, we can try it.' Mr. C replied. And so I embarked on my quest. I researched different workshops on the internet, learning the information myself at first. Then, I transitioned into creating workshops of my own, making sure that the information was easy to understand for even a beginner. I was exhausted; my first workshop took 16 cumulative hours to create."""
# 
char_sec_re = character_all_section(text)

print(char_sec_re)

one_ps_char_desc = char_sec_re[0] #튜플에서 첫번재 인댁스 값 가져오기 : Character Descriptiveness



## 1000명 데이터의 각 값(char_desc_mean)의 평균 값 전달.  list type ex) [98, 876, 45, 67]
char_desc_mean = [72.4, 321.4, 148.8, 93.6]
ideal_mean = char_desc_mean[0] #첫번째 값을 가져옴

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
compare = (one_ps_char_desc + ideal_mean)/7
print('compare :', compare)



if one_ps_char_desc > ideal_mean: # 개인점수가 평균보다 클 경우는 overboard
    if cal_abs > compare: # 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
        print("Overboard")
        result = 2
    else: #차이가 많이 안나면
        print("Ideal")
        result = 1
    
elif one_ps_char_desc < ideal_mean: # 개인점수가 평균보다 작을 경우 lacking
    if cal_abs > compare: #차이가 많이나면 # 개인점수가  평균보다 작을 경우 Lacking이고 
        print("Lacking")
        result = 0
    else: #차이가 많이 안나면
        print ("Ideal")
        result = 1
        
else:
    print("Ideal")
    result = 1