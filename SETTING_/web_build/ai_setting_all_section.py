# Setting 전체 분석 코드

# 총 12개의 분석이 가능함

############# 실행함수 ##############
# ai_setting_overallratio(input_text)


# 최종결과(12개임) :  [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 3.91]

###  lacking : 0, ideal: 1, overboard : 2  를 의미함 ###



# 1) setting_indicators_fin : 1 lacking

#### 구간별 setting indicator

# 2) intro_re : 1 ideal 
# 3) body_re 0 lacking
# 4) conclusion_re 1 ideal

# 전체 입력 데이터셋의  plance nouns 비율
# 5) place_nouns_ratio 0 lacking

#### 구간별 장소 명사(Place Nuns) 계산
# 6) place_nouns_intro 0 lacking
# 7) place_nouns_body 0 lacking
# 8) place_nouns_conclusion 1 ideal

#### 전체 Setting Descriptiveness 
# 9) re_char_desp_fin 1 ideal

#### 구간별 Setting Descriptiveness 
# 10) re_char_desp_fin_intro 1 ideal 
# 11) re_char_desp_fin_body 1 ideal 
# 12) re_char_desp_fin_conclusion 1 ideal

# 맨 마지막 값은 '3.91'은 5점 만점중 평균비교 종합계산한 점수임


import numpy as np
import pandas as pd
import gensim
import nltk
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
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

nltk.download('averaged_perceptron_tagger')

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F
import pickle

#cuda 메모리에 여유를 주기 위해서 잠시 딜레이를 시키자
import time



def setting_analy_all(text):

    def Setting_analysis(text):

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
        location_list = ['above', 'behind','below','beside','by','in','inside','near',
                        'on','over','through','about', 'above', 'along', 'approach', 'approximate', 'aside', 'behind',
                        'below', 'buttocks', 'by', 'cheeseparing', 'complete', 'dear', 'done', 'downstairs',
                        'in', 'inch', 'indiana', 'indium', 'inside', 'inwardly', 'near', 'on', 'over', 'through', 'under']
        
        time_list = ['after', 'before','by','during','from','on','past','since','through','to','until','upon',
                    'ahead', 'along', 'aside', 'by', 'done', 'earlier', 'on', 'past', 'subsequently']
        
        movement_list = ['against','along','down','from','into','off','on','onto','out of','toward','up','upon'
                        'along', 'astir', 'away', 'depressed', 'devour', 'down', 'gloomy', 'improving', 'murder',
                        'on', 'polish', 'up']
        
        palce_terrain_type_list = ['wood', 'forest', 'copse', 'bush', 'trees', 'stand',
                                    'swamp', 'marsh', 'wetland', 'fen', 'bog', 'moor', 'heath', 'fells', 'morass',
                                    'jungle', 'rainforest', 'cloud forest','plains', 'fields', 'grass', 'grassland', 
                                    'savannah', 'flood plain', 'flats', 'prairie','tundra', 'iceberg', 'glacier', 
                                    'snowfields','hills', 'highland,' 'heights', 'plateau', 'badland', 'kame', 'shield',
                                    'downs', 'downland', 'ridge', 'ridgeline','hollow,' 'valley',' vale','glen', 'dell',
                                    'mountain', 'peak', 'summit', 'rise', 'pass', 'notch', 'crown', 'mount', 'switchback',
                                    'furth','canyon', 'cliff', 'bluff,' 'ravine', 'gully', 'gulch', 'gorge',
                                    'desert', 'scrub', 'waste', 'wasteland', 'sands', 'dunes',
                                    'volcano', 'crater', 'cone', 'geyser', 'lava fields','abandon',
    'acme', 'advance', 'afforest', 'airfield', 'apartment', 'arise', 'ascend', 'ascent', 'authorize', 'backbone',
    'backing', 'bandstand', 'barren', 'base', 'batch', 'battlefield', 'bill', 'bog', 'brush', 'bush', 'bye', 'cancel',
    'canyon', 'carapace', 'cliff', 'climb', 'communicate', 'complain', 'cone', 'consume', 'corner', 'crack', 'crater',
    'crown', 'defect','defile', 'dell', 'deluge', 'denounce', 'desert', 'devour', 'die', 'digest', 'discipline',
    'down', 'dun', 'dune', 'eatage', 'elapse', 'emanation', 'esophagus', 'evanesce', 'exceed', 'excrete', 'extremum',
    'fall', 'fell', 'fen', 'field', 'fields', 'flat','flatcar', 'flats', 'flower', 'fly', 'forest', 'geyser', 'glacier',
    'glen', 'godforsaken', 'gorge', 'grass', 'grassland', 'guide', 'gulch', 'gully','happen', 'harbor', 'heath', 'heighten',
    'hide', 'hill', 'iceberg', 'jungle', 'knit','league', 'legislate','lift', 'littoral','marsh', 'mire', 'moor', 'mound',
    'mount', 'mountain', 'neutralize', 'notch', 'originate', 'pas', 'pass', 'passing','pate', 'peak', 'pennant', 'plain',
    'point', 'polish', 'pot', 'prairie', 'rack', 'raise', 'rebel', 'resist', 'resurrect','ride', 'ridge', 'rise', 'run',
    'sand', 'sandpaper', 'savanna', 'savannah','scrub', 'shield', 'shrub', 'sink', 'snowfield', 'spend', 'sphere', 'stall',
    'stand', 'summit', 'supergrass', 'surface', 'swamp', 'tableland', 'thriftlessness', 'torment', 'tree', 'tundra', 'upgrde',
    'vent', 'vertex', 'volcano', 'waste', 'wax','wetland', 'wood', 'woodwind']
        
        water_list = ['ocean', 'sea', 'coast', 'beach', 'shore', 'strand','bay', 'port', 'harbour', 'fjord', 'vike',
                    'cove', 'shoals', 'lagoon', 'firth', 'bight', 'sound', 'strait', 'gulf', 'inlet', 'loch', 
                    'bayou','dock', 'pier', 'anchorage', 'jetty', 'wharf', 'marina', 'landing', 'mooring', 'berth', 
                    'quay', 'staith','river', 'stream', 'creek', 'brook', 'waterway', 'rill','delta', 'bank', 'runoff',
                    'channel', 'bend', 'meander', 'backwater','lake', 'pool', 'pond', 'dugout', 'fountain', 'spring', 
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

        building_facilities_list = ['temple', 'shrine', 'church', 'cathedral', 'tabernacle', 'ark', 'sanctum', 'parish', 
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
        
        # for i in filtered_setting_text__:
        #     ext_setting_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
        setting_total_count = len(filtered_setting_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현 수
        setting_count_ = len(filtered_setting_text__) #중복제거된 setting표현 총 수
            
        result_setting_words_ratio = round(setting_total_count/total_words * 100, 2)
        return result_setting_words_ratio
        #return result_setting_words_ratio, total_sentences, total_words, setting_total_count, setting_count_, ext_setting_sim_words_key


    ##########################################################
    # 650단어에서 또는 전체 단어에서 단락별 셋팅단어 활용 수 분석
    # 30% intro 50% body 20% conclusion
    ##########################################################

    # 전체 입력 문장을 30% intro 50% body 20% conclusion 구분한다.

    #def paragraph_divide_ratio(text):

    essay_input_corpus = str(text) #문장입력
    essay_input_corpus = essay_input_corpus.lower()#소문자 변환

    sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화

    # 총 문장수 계산
    total_sentences = len(sentences) # 토큰으로 처리된 총 문장 수

    # 비율계산 시작
    intro_n = round(len(sentences)*0.3) # 30% 만 계산하기, 소수점이하는 반올림
    body_n = round(len(sentences)*0.5) # 50% 만 계산하기, 소수점이하는 반올림
    conclusion_n = round(len(sentences)*0.2) # 20% 만 계산하기, 소수점이하는 반올림

    #데이터셋 비율분할 완료
    intro = sentences[:intro_n]
    body = sentences[intro_n:body_n]
    conclusion = sentences[body_n:]

    ################### - start -######################
    # 구간별 셋팅 인디케이터 intro - body - conclusion  
    ###################################################

    def indications_setting(text):

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
        location_list = ['above', 'behind','below','beside','by','in','inside','near',
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
                    'channel', 'bend', 'meander', 'backwater','lake', 'pool', 'pond', 'dugout', 'fountain', 'spring', 
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

        building_facilities_list = ['temple', 'shrine', 'church', 'cathedral', 'tabernacle', 'ark', 'sanctum', 'parish', 
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

        # # load
        # with open('glove_vectors.pickle', 'rb') as f:
        #     glove_vectors = pickle.load(f)

        # filt_wds_list = []
        # for i in setting_words_filter_list:
        #     ext_setting_sim_words_key = glove_vectors.most_similar(i)#모델적용>>>>>>>>>>>>>  이 부분을 glove_vectors.most_similar(i) 로 바꿀 것 
        #     filt_wds_list.append(ext_setting_sim_words_key)

        # setting_words_filter_list_ = sum(filt_wds_list, [])

        
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
        

        
        setting_total_count = len(filtered_setting_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현 수
        setting_count_ = len(filtered_setting_text__) #중복제거된 setting표현 총 수
            
        result_setting_words_ratio = round(setting_total_count/total_words * 100, 2)
        return result_setting_words_ratio
        #return result_setting_words_ratio, total_sentences, total_words, setting_total_count, setting_count_, ext_setting_sim_words_key

    ################### - end -######################
    #################################################
    # 구간별 셋팅 intro - body - conclusion   계산 시작
    #################################################

    # setting indicators 
    setting_indicators = indications_setting(text)

    # intro 의 setting 분석
    intro_re = indications_setting(intro)

    # body 의 setting 분석
    body_re = indications_setting(body)

    # intro 의 setting 분석
    conclusion_re = indications_setting(conclusion)


    ################# - start - #######################
    ############ PLACE NOUNS  계산 시작 ###############
    ###################################################
    def placenouns_analysis(text):

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
        location_list = ['above', 'behind','below','beside','by','in','inside','near',
                        'on','over','through']
        #time_list = ['after', 'before','by','during','from','on','past','since','through','to','until','upon']
        
        #movement_list = ['against','along','down','from','into','off','on','onto','out of','toward','up','upon']
        
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
                    'channel', 'bend', 'meander', 'backwater','lake', 'pool', 'pond', 'dugout', 'fountain', 'spring', 
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

        building_facilities_list = ['temple', 'shrine', 'church', 'cathedral', 'tabernacle', 'ark', 'sanctum', 'parish', 
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
        
        
        setting_words_filter_list = location_list + palce_terrain_type_list + water_list + outdoor_places_list + underground_list + underground_list + living_places_list + building_facilities_list + architecture_list

        
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
        
        # for i in filtered_setting_text__:
        #     ext_setting_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
        setting_total_count = len(filtered_setting_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현 수
        setting_count_ = len(filtered_setting_text__) #중복제거된 setting표현 총 수
            
        result_setting_words_ratio = round(setting_total_count/total_words * 100, 2)
        return result_setting_words_ratio
        #return result_setting_words_ratio, total_sentences, total_words, setting_total_count, setting_count_, ext_setting_sim_words_key


    ################# - end - #######################

    # 전체 입력 데이터셋의 PLACE NOUNS 비율 계산
    place_nouns_ratio = placenouns_analysis(text)

    #intro - PLACE NOUNS 계산
    place_nouns_intro = placenouns_analysis(intro)

    #body - PLACE NOUNS 계산
    place_nouns_body = placenouns_analysis(body)

    #conclusion - PLACE NOUNS 계산
    place_nouns_conclusion = placenouns_analysis(conclusion)


    ################################# - start - ######################################
    ######################### Setting Descriptiveness ################################
    ##################################################################################
    ## 1)장소, 시간, 공간 단어가 들어있는 문장 추출하기
    ## 2)추출한 문장에서 showing이 얼마나 들어갔는지 비율 계산하기

    def setting_descriptiveness(each_input_text):

        input_text = each_input_text

        def findSentence(input_sentence):
            result = []

            data = str(input_sentence)
            #data = input_sentence.splitlines()
            
            
            #setting을 표현하는 단어들을 리스트에 넣어서 필터로 만들고
            location_list = ['above', 'behind','below','beside','by','in','inside','near',
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
                        'channel', 'bend', 'meander', 'backwater','lake', 'pool', 'pond', 'dugout', 'fountain', 'spring', 
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

            building_facilities_list = ['temple', 'shrine', 'church', 'cathedral', 'tabernacle', 'ark', 'sanctum', 'parish', 
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
            
            
            findText = location_list + time_list + movement_list + palce_terrain_type_list + water_list + outdoor_places_list + underground_list + underground_list + living_places_list + building_facilities_list + architecture_list


            sentences = data.split(".")
            
            for sentence in sentences:
                for item in findText:
                    if item in sentence:
                        result.append(sentence)

            return result


        ################################# - end - ######################################


        input_sent_included_character = findSentence(input_text) 
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
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            model = BertForSequenceClassification.from_pretrained('bert-base-cased')
            model.to(device)


            model = torch.load("/Users/kimkwangil/Documents/001_ESSAYFITAI/essayfitaiproject_2020_12_22 2/essayai/data/model.pt", map_location=torch.device('cpu'))
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


        #st_re = showtell_classfy(input_text) # 캐릭터거 포함된 문장(전처리 완료된) 입력
        st_re = showtell_classfy(str(input_sent_chr)) # 캐릭터거 포함된 문장(전처리 완료된) 입력

        df = pd.DataFrame(st_re)
        df_ = df[0::2] # 글만 추출
        df_label = df[1::2] # 레이블만 추출
        df_.reset_index(drop=True, inplace=True) #데이터를 합치기 위해서 초기화
        df_label.reset_index(drop=True, inplace=True)
        df_result = pd.concat([df_,df_label],axis=1) # 값 합치기
        df_result.columns = ['sentence','show/tell']
        df_fin = df_result['show/tell'].value_counts(normalize=True)
        list(df_fin)

        #이건 character 설명 문장에 showing이 얼마나 들어가 있는지… 평균대비

        showing_sentence_with_settig_descrip = max(round(df_fin*100))

        print ('Setting Ratio in Sentence : ', showing_sentence_with_settig_descrip)

        return showing_sentence_with_settig_descrip


    #################################################################################
    # CHARACTER DESCRIPTIVENESS 값 계산시작!

    re_char_desp_fin = setting_descriptiveness(text)
    re_char_desp_fin_intro = setting_descriptiveness(intro)
    re_char_desp_fin_body = setting_descriptiveness(body)
    re_char_desp_fin_conclusion = setting_descriptiveness(conclusion)

    ### 최종 결과 계산!!!! 끝!!!
    return setting_indicators, intro_re, body_re, conclusion_re, place_nouns_ratio, place_nouns_intro, place_nouns_body, place_nouns_conclusion, re_char_desp_fin, re_char_desp_fin_intro, re_char_desp_fin_body, re_char_desp_fin_conclusion

########### return 값 설명 ###########

## 계산결과 ex) RESULT : (6.34, 1.84, 8.85, 2.47, 1.12, 0.61, 2.9, 77.0, 69.0, 45.0, 86.0)

#전체 setting_indicators
# setting_indicators

# 구간별 setting indicator
# intro_re
# body_re
# conclusion_re

# 전체 입력 데이터셋의  plance nouns 비율
# place_nouns_ratio

# 구간별 장소 명사(Place Nuns) 계산
# place_nouns_intro
# place_nouns_body
# place_nouns_conclusion

# 전체 Setting Descriptiveness 
# re_char_desp_fin

# 구간별 Setting Descriptiveness 
# re_char_desp_fin_intro
# re_char_desp_fin_body
# re_char_desp_fin_conclusion

def ai_setting_overallratio(input_text):

    setting_one_ps = setting_analy_all(input_text)

    print("1명의 에세이 결과 계산점수 :", setting_one_ps)

    # 위에서 계산한 총 4개의 값을 개인, 그룹의 값과 비교하여 lacking, ideal, overboard 계산

    # 개인에세이 값 계산 11가지 결과 추출 >>>>> personal_value 로 입력됨
    setting_indicators = setting_one_ps[0]
    intro_re = setting_one_ps[1]
    body_re = setting_one_ps[2]
    conclusion_re = setting_one_ps[3]
    place_nouns_ratio = setting_one_ps[4]
    place_nouns_intro = setting_one_ps[5]
    place_nouns_body = setting_one_ps[6]
    place_nouns_conclusion = setting_one_ps[7]
    re_char_desp_fin = setting_one_ps[8]
    re_char_desp_fin_intro = setting_one_ps[9]
    re_char_desp_fin_body = setting_one_ps[10]
    re_char_desp_fin_conclusion = setting_one_ps[11]
    
    ## 1000명 데이터의 평균값 (이미 계산한 결과를 반영하여 적용: 고정값임)
    set_result_1000 = [7.8, 7.8, 8.20, 9.29, 3.965, 3.1775, 3.0, 3.8, 87.75, 93.75, 90.0, 94.25]

    #1000명의 학생에 에세이 값을 가져온다.
    setting_indicators_mean = set_result_1000[0]
    intro_re_mean = set_result_1000[1]
    body_re_mean = set_result_1000[2]
    conclusion_re_mean = set_result_1000[3]
    place_nouns_ratio_mean = set_result_1000[4]
    place_nouns_intro_mean = set_result_1000[5]
    place_nouns_body_mean = set_result_1000[6]
    place_nouns_conclusion_mean = set_result_1000[7]
    re_char_desp_fin_mean = set_result_1000[8]
    re_char_desp_fin_intro_mean = set_result_1000[9]
    re_char_desp_fin_body_mean = set_result_1000[10]
    re_char_desp_fin_conclusion_mean = set_result_1000[11]

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


    #종합계산시작 lackigIdealOverboard(group_mean, personal_value)
    setting_indicators_fin = lackigIdealOverboard(setting_indicators_mean, setting_indicators)
    intro_re_fin = lackigIdealOverboard(intro_re_mean, intro_re)
    body_re_fin = lackigIdealOverboard(body_re_mean, body_re)
    conclusion_re_fin = lackigIdealOverboard(conclusion_re_mean, conclusion_re)
    place_nouns_ratio_fin = lackigIdealOverboard(place_nouns_ratio_mean, place_nouns_ratio)
    place_nouns_intro_fin = lackigIdealOverboard(place_nouns_intro_mean, place_nouns_intro)
    place_nouns_body_fin = lackigIdealOverboard(place_nouns_body_mean, place_nouns_body)
    place_nouns_conclusion_fin = lackigIdealOverboard(place_nouns_conclusion_mean, place_nouns_conclusion)
    re_char_desp_fin_fin = lackigIdealOverboard(re_char_desp_fin_mean, re_char_desp_fin)
    re_char_desp_fin_intro_fin = lackigIdealOverboard(re_char_desp_fin_intro_mean, re_char_desp_fin_intro)
    re_char_desp_fin_body_fin = lackigIdealOverboard(re_char_desp_fin_body_mean, re_char_desp_fin_body)
    re_char_desp_fin_conclusione_fin = lackigIdealOverboard(re_char_desp_fin_conclusion_mean, re_char_desp_fin_conclusion)

    fin_result = [setting_indicators_fin, intro_re_fin, body_re_fin, conclusion_re_fin, place_nouns_ratio_fin, place_nouns_intro_fin, place_nouns_body_fin,
                    place_nouns_conclusion_fin, re_char_desp_fin_fin, re_char_desp_fin_intro_fin, re_char_desp_fin_body_fin, re_char_desp_fin_conclusione_fin]


    each_fin_result = [fin_result[0][0], fin_result[1][0], fin_result[2][0], fin_result[3][0],
                    fin_result[4][0],fin_result[5][0], fin_result[6][0], fin_result[7][0], fin_result[8][0],
                    fin_result[9][0],fin_result[10][0], fin_result[11][0]]


    overall_setting_rating = [round((fin_result[0][1]
                           +fin_result[1][1] 
                           +fin_result[2][1]
                           +fin_result[3][1]
                           +fin_result[4][1]
                           +fin_result[5][1]
                           +fin_result[6][1]
                           +fin_result[7][1]
                           +fin_result[8][1]
                           +fin_result[9][1]
                           +fin_result[10][1]
                           +fin_result[11][1])/12, 2)]

    result_final = each_fin_result + overall_setting_rating
    
    
    
    data  = {
        
        "Indicators_ratio": result_final[0],
        
        "Indicators_intro_re": result_final[1],
        "Indicators_body_re": result_final[2],
        "Indicators_conclusion_re": result_final[3],
        "place_nouns_ratio": result_final[4],
        
        "place_nouns_intro": result_final[5],
        "place_nouns_body": result_final[6],
        "place_nouns_conclusion": result_final[7],
        
        "Descriptiveness": result_final[8],
        
        "desp_fin_intro": result_final[9],
        "desp_fin_body": result_final[10],
        "desp_fin_conclusion": result_final[11],
        
        "avg_setting": result_final[12]        
        
        
        
    }

    return data



#################### 테스트~~~~!!!! ################### 테스트하고 주석처리하시오. 

input_text= """A window into the soul.For most people, this would be the eyes. The eyes cannot lie; they often tell more about a person's emotions than their words. What distinguishes a fake smile from a genuine one? The eyes. What shows sadness? The eyes. What gives away a liar? The eyes.But are the eyes the only window into the soul?Recently, I began painting with watercolors. With watercolors, there is no turning back: if one section is too dark, it is nearly impossible to lighten the area again. Every stroke must be done purposefully, every color mixed to its exact value.I laid my materials before me, preparing myself for the worst. I checked my list of supplies, making sure my setup was perfect.I wet my brush, dipped it into some yellow ochre, and dabbed off the excess paint. Too little water on my brush. I dipped my brush back into my trusty water jar; the colors swirled beautifully, forming an abstract art piece before my eyes. \u2014It's a shame that I couldn't appreciate it.I continued mixing colors to their exact value. More alizarin crimson. More water. More yellow ochre. Less water. More phthalo blue. The cycle continued. Eventually, I was satisfied. The colors looked good, there was enough contrast between facial features, and the watercolors stayed inside the lines.Craving feedback, I posted my art to Snapchat. I got a few messages such as 'wow' and 'pretty,' but one message stood out. 'You were anxious with this one, huh? Anyways, love the hair!'I was caught off guard. Was it a lucky guess? Did they know something I didn't? I immediately responded: 'Haha, how could you tell?' No response.What I didn't know at the time was that my response would come a few months later while babysitting. Since the girl I was babysitting loved art, I took out some Crayola watercolors and some watercolor paper for her to play with. After I went to the bathroom and came back, the watercolors were doused with water. 'You were impatient with this one, huh? Anyways, love the little dog you drew!'The little girl looked up at me, confused. 'How could you tell?' 'You used a lot of water for a brighter color, but you couldn't wait for it to slowly soak in.''Oh.'Now, I would be lying if I said I realized the connection between the two events immediately.Instead, I made the connection when I decided to sit down one day and objectively critique my art. The piece that I once loved now seemed like a nervous wreck: the paper was overworked, the brushstrokes were undecided, the facial features blended together, and each drop of water was bound inside the lines as if it was a prisoner in a cage.From then on, I started noticing pieces of personality in additional creations surrounding me: website designs, solutions to math problems, code written for class, and even the preparation of a meal.When I peer around at people's projects during Code Club, I notice the clear differences between their code. Some people break it up by commenting in every possible section. Others breeze through the project, not caring to comment or organize their code. I could also see clear differences in personalities when our club members began coding the Arduino for the first time. Some followed the tutorials to the letter, while others immediately started experimenting with different colored LEDs and ways of wiring the circuit.It became clear to me that, as humans, we leave pieces of our souls in everything we do, more than we intend to. If we entertain this thought, perhaps the key to better understanding others around us is simply noticing the subtler clues under our noses?Perhaps there are endless windows to the soul, and we simply need to peer through them. I shakily rose my hand. 'We should create workshops of our own,' I suggested.I got a few strange looks. 'It's a good idea, but it's too much work.' 'We just don't have enough free time to make it work.' 'Maybe we could, but I don't know how to make workshops.' My suggestion was shot down. I shuffled in my seat. 'I could make them.' A few people stared at me in disbelief. I glanced over at the club advisor, Mr. C, nervous to hear his response.'If you're willing to take on the work, we can try it.' Mr. C replied. And so I embarked on my quest. I researched different workshops on the internet, learning the information myself at first. Then, I transitioned into creating workshops of my own, making sure that the information was easy to understand for even a beginner. I was exhausted; my first workshop took 16 cumulative hours to create."""


print("#########################################################################")
print("최종결과 : ", ai_setting_overallratio(input_text))
# 최종결과 :  {'Indicators_ratio': 1, 'Indicators_intro_re': 1, 'Indicators_body_re': 0, 'Indicators_conclusion_re': 1, 'place_nouns_ratio': 0, 'place_nouns_intro': 0, 'place_nouns_body': 0, 'place_nouns_conclusion': 1, 'Descriptiveness': 1, 'desp_fin_intro': 1, 'desp_fin_body': 1, 'desp_fin_conclusion': 1, 'avg_setting': 4.0}



# Setting 전체 분석 코드

# 총 12개의 분석이 가능함

############# 실행함수 ##############
# ai_setting_overallratio(input_text)


# 최종결과(12개임) :  [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 3.91]

###  lacking : 0, ideal: 1, overboard : 2  를 의미함 ###



# 1) setting_indicators_fin : 1 lacking

#### 구간별 setting indicator

# 2) intro_re : 1 ideal
# 3) body_re 0 lacking
# 4) conclusion_re 1 ideal

# 전체 입력 데이터셋의  plance nouns 비율
# 5) place_nouns_ratio 0 lacking

#### 구간별 장소 명사(Place Nuns) 계산
# 6) place_nouns_intro 0 lacking
# 7) place_nouns_body 0 lacking
# 8) place_nouns_conclusion 1 ideal

#### 전체 Setting Descriptiveness 
# 9) re_char_desp_fin 1 ideal

#### 구간별 Setting Descriptiveness 
# 10) re_char_desp_fin_intro 1 ideal 
# 11) re_char_desp_fin_body 1 ideal 
# 12) re_char_desp_fin_conclusion 1 ideal

# 맨 마지막 값은 '3.91'은 5점 만점중 평균비교 종합계산한 점수임