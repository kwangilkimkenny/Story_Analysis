#conflict
import pickle
import nltk

# 다운로드 이미 완료, 실행시 사용하지 않음
# punkt = nltk.download('punkt')
# with open('nltk_punkt.pickle', 'wb') as f:
#     pickle.dump(punkt, f, pickle.HIGHEST_PROTOCOL)

with open('./data_accepted_st/nltk_punkt.pickle', 'rb') as f:
    punkt = pickle.load(f)

# 다운로드 이미 완료, 실행시 사용하지 않음
# vader_lexicon = nltk.download('vader_lexicon')
# with open('vader_lexicon.pickle', 'wb') as f:
#     pickle.dump(punkt, f, pickle.HIGHEST_PROTOCOL)

with open('./data_accepted_st/vader_lexicon.pickle', 'rb') as f:
    vader_lexicon = pickle.load(f)

# 다운로드 이미 완료, 실행시 사용하지 않음
# averaged_perceptron_tagger  = nltk.download('averaged_perceptron_tagger')
# with open('averaged_perceptron_tagger.pickle', 'wb') as f:
#     pickle.dump(punkt, f, pickle.HIGHEST_PROTOCOL)
    
with open('./data_accepted_st/averaged_perceptron_tagger.pickle', 'rb') as f:
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
    
with open('./data_accepted_st/stopwords.pickle', 'rb') as f:
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
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from pprint import pprint

tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-group")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-group")

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
        #print('ext_sim_words_key : ', ext_sim_words_key)
        

        #return result_char_ratio, total_sentences, total_words, char_total_count, char_count_, ext_sim_words_key
        return ext_sim_words_key, total_words



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
    #print('token_list_str:', token_list_str)
    confict_words_list_basic = ['clash', 'incompatible', 'inconsistent', 'incongruous', 'opposition', 'variance','vary', 'odds', 
                            'differ', 'diverge', 'disagree', 'contrast', 'collide', 'contradictory', 'incompatible', 'conflict',
                            'inconsistent','irreconcilable','incongruous','contrary','opposite','opposing','opposed', 'fight',
                            'antithetical','clashing','discordant','differing','different','divergent','discrepant', 'beef', 'bone to pick',
                            'varying','disagreeing','contrasting','at odds','in opposition','at variance', 'bone of contention',
                            'battle', 'competition', 'combat', 'rivalry', 'strife', 'struggle', 'war', 'collision',
                            'contention', 'contest', 'emulation', 'encounter', 'engagement', 'fracas', 'fray', 'set-to',
                            'striving', 'tug-of-war', 'conflicted', 'conflicting', 'conflicts', 'disagreement','contrariety',
                            'friction', 'enmity', 'dissension', 'incongruity', 'rancor', 'resistance', 'hostility', 'hatred',
                            'discord', 'debate', 'controversy', 'dispute', 'agitation','matter','matter at hand','problem',
                            'point in question', 'question', 'dispute', 'issue', 'sore point', 'tender spot', 'quarrel', 'discord',
                            'greek','grain','gain','wage','dialogue','fast–moving','feud','obverse','flame','therapy','blood','fuel','distraught',
                            'conflicting','ado','duel','interference','match','insurgency','philistine','go','contradiction','settlement','outrage',
                            'add','uprising','information','mineral','robber','fratricidal','horn','rudeness','agrarian','third','loggerhead',
                            'embroil','yitzhak','mellay','contention','cop','contravention','private','sectional','hot','absurdism','fire','finish',
                            'social','antinomy','recuse','science','brink','ombudsman','stour','resolve','adversary','strained','anti-democratic',
                            'coordinate','discord','challenge','ambivalence','dove','vanir','someone','bigot','vantage','meet','reˈpression',
                            'belli','feudal','jar','combatant','conversion','aufhebung','identity','along','bygone','lion','mus','field',
                            'role-playing','war','course','scrape','support','v-e',"jusqu'auboutisme",'class','war','gas','embed','conflictive',
                            'mauriac','tangle','disharmony','discrepancy','vidal','tooth','militate','front','bloodshed','coping','state','cold',
                            'inharmonious','action',"jusqu'au",'disagreement','aggress','corporatism','suez','alarum','dissension','escalation',
                            'twivolution','internationalism','logomachy','manichaeism','repress','crisis','violent','ragnarok','militation','heal',
                            'twilight','ripe','side','contest','russian','list','shooting','arbitrate','militant','head-to-head','parapraxis','sustained',
                            'cool','comedy','dramatic','ossetia','shamir','triumph','school','activity','militarize','biffo','apple','peace',
                            'coexist','western','fighting','victor','turbulence','bloodless','agon','post-war','polemical','horn','stomach',
                            'cliff–hang','john','ally','stand','glance','casus','vehemently','dagger','perennial','engagement','repugnance',
                            'infighting','bunche','abkhazia','interest','go-slow',"jusqu'auboutiste",'materialism','full-fledged','embroiled',
                            'dragon','peacenik','arena','hold','trench','fray','state','incompatible','conflagration','compatible','truce',
                            'cyberwar','lose','decide','reluctant','riot','stormy','brush','part','head','politics','warrior','infiltrate','melee',
                            'additional','set','revolt','complex','face','psychogenic','hound','tussle','laxism','enemy','multinational','battleground',
                            'chairman','renvoi','day','day','janjaweed','chuff','troublemaker','tug-of-love','ammunition','truculent','mercenary',
                            'quarrel','anal-retentive','forearm','board','ralph','heroic','drama','commission','mechanism','marxism','join','windmill',
                            'conquest','cry','internecive','clash','pea','psychoneurosis','shower','displacement','pacific','buffer','continue','foe',
                            'self-consistent','federal','irrepressible','armageddon','sword','abreact','dialectical','butt','class–angle','clark',
                            'eastern','repugnant','karen','dynamite','diamond','carthage','coal-blower','hatchet','darwinism','red','internally',
                            'discordant','sectarian','system','overcome','casevac','stricken','collide','conflict','neutral','cross-current','civil',
                            'falklands','seven','loss','cleopatra','ireland','toll','gridlock','shatter','chemical','engage','sideways','escalate',
                            'confrontation','disengage','arbiter','instigate','henry','tragedy','retaliation','bourgeoisie','conflict-free','peripeteia',
                            'neutralism','disagree','farouk','wayne','incendiary','crosswise','rage','anal','culture','protagonist','warpath','foul','measure',
                            'synthesis','intervene','throat','battlefield','belligerency','enter','neurasthenia','psychosomatic','coopetition','gird','theory',
                            'encounter','nerve','heretic','turbulent','motorize','friction','ground','force','strife','vanquish','advertisement','mark',
                            'faction','displaced','reckoning','undivided','duke','internecion','eris','crush','bloodletting','dogfight','firework','line',
                            'altercation','footfight','zone','god','polarize','priština','sciamachy','wartime','insurrection','ulster','high','racial',
                            'apocalypse','peacemaker','belligerent','preemption','claw','play','psychomachy','transracial','bitter','re-escalation','conflicted',
                            'hostage','race','gulf','peacefulness','dog','note','mobilization','compliance','trounce','chiefly','pacification','eristic',
                            'interactionism','international','de-escalate','warfare','contravene','give','let','chorus','cognitive','ring','front','latvia',
                            'offense','trust','intensification','apocalyptic','arm','bout','mix–up','northern','saul','pacifist','revolutionary','set-to',
                            'communal','containment','internecine','peaceful','contrary','rebellion','psychomachia','rencounter','face–off','no-fly','middleman',
                            'conquer','uncontroversial','officer','recusation','petrel','beat','darfur','communism','mujaheddin','game','get','adjudicate','engaged',
                            'friendly','jotun','odds','contradict','reign','struggle','civil','lawful','censure','aceh','serbia','charles','boilover','hare',
                            'rising','climax','disorder','spoil','embattle','cold','jostle','parley','carry','dispute','umpire','dissonant','gide','precipitate',
                            'armed','compatibility','oppose','abhorrent','afflict','trincomalee','repugn','come','couple','oppugn','peaceable','blind','contretemps'
                            'debate','agree','american','camp','run','kulturkampf','dialectic','top','head-on','reunification','inferiority','trouble','polemology',
                            'henry','invasion','contradictory','battle','lock','temporize','diplomat','rude','law','blitz','hunt','impi','upheaval','bovarism',
                            'great','torn','words','re-escalate','bury','obstacle','shell','medal','disunity','dissonance','wesley','block','agonistic','thucydides',
                            'skirmish','north','strive','cockpit','concordance','idealist','hostility','idp','biowar','lubricant','morality','collision','mahabharata',
                            'defence','variance','boot','afoul','combat','interpersonal','foreign','internal','embattled','opposition','south','underdog','xxii',
                            'opponent','adversarial','disown','fight','firebrand','military','belligerence','resolution','hamburger','militancy','mano','stalemate',
                            'rapprochement','fall','dickin','reconciliation',"years'",'problem','spanish']

    confict_words_list = confict_words_list_basic + conflict_sim_words_ratio_result[0] #유사단어를 계산결과 반영!
    #중복제거
    confict_words_list_set = set(confict_words_list)
    #print('confict_words_list:', confict_words_list_set)
    
    # 문장에 들어있는 추출된 conflict 단어들 : count_conflict_list ==================> conflict 단어가 없음(겹치는 단어 없나?)
    count_conflict_list = []
    for ittm in confict_words_list_set:
        if ittm in token_list_str_set:
            count_conflict_list.append(ittm)
            
    #print('문장에 들어있는 추출된 conflict 단어들:', count_conflict_list)
    
    # 전체문장에 들어있는 conflict 단어 수
    nums_conflict_words =  len(count_conflict_list)
    #print('전체문장에 들어있는 conflict 단어 수:', nums_conflict_words)

    #conflict_sim_words_ratio_result[1] : 총 단어 수
    return nums_conflict_words, conflict_sim_words_ratio_result[1]



def get_conflict_words():

    ratio_score_cnt = []
    mean_essay_words_nums = []
    path = "./data/accepted_data/ps_essay_evaluated.csv"
    data = pd.read_csv(path)
    #Score를 인덱스로 변환하여 데이터 찾아보기
    data.set_index('Score', inplace=True)
    for i in tqdm(data.index):
        if i is not None and i >= 4:
            get_essay = data.loc[i, 'Essay']

            input_ps_essay = get_essay
            result = ai_plot_conf(str(input_ps_essay))
            ratio_score_cnt.append(result[0])
            mean_essay_words_nums.append(result[1])


    #print('emotion_counter:', emotion_counter)
    # e_re = [y for x in ratio_score_cnt for y in x]
    # # 중복감성 추출
    # total_count = {}
    # for i in e_re:
    #     try: total_count[i] += 1
    #     except: total_count[i]=1

    # 전체문장에 들어있는 conflict 단어의 사용 비율
    accepted_mean = round(sum(ratio_score_cnt) / len(ratio_score_cnt), 1)
    print('accepted_mean:', accepted_mean)
    # 전체 문장에서의 컨플릭 단어 평균값
    mean_wd_essay = round(sum(mean_essay_words_nums) / len(mean_essay_words_nums), 1)
    print('mean_wd_essay:', mean_wd_essay)



    #전체 문장에서 컨플릭 단어의 포함 비율
    result = round(accepted_mean / mean_wd_essay * 100 , 1)

    data = {
        '합격한 학생들의 전체 문장에서 Conflict Words 평균 활용 비율' : result, 
        '합격한 학생들의 에세이에 포함된 Conflict words 평수 수' : accepted_mean, 
        '함격한 학생들의 에세이에 적용된 평균 단어 수' : mean_wd_essay
    }


    return data 


print('Result :', get_conflict_words())