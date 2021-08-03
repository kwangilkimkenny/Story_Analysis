# upgrade... 2021_07_27

# 아나콘다 가상환경 office:  py37TF2
# home : py37Keras

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
import spacy

nlp = spacy.load('en_core_web_lg')


##########  key_value_print
##########  key: value 형식으로 나뉨 
def key_value_print (dictonrytemp) : 
    print("#"*100)
    for key in dictonrytemp.keys() : 
        print(key,": ",dictonrytemp[key])
        print()
    print("#"*100)


# 시간, 공간, 장소를 알려주는 단어 추출하여 카운트
def find_setting_words(text):
    # Create Doc object
    doc2 = nlp(text)
    
    setting_list = []
    # Identify by label FAC(building etc), GPE(countries, cities..), LOC(locaton), TIME
    fac_r = [ent.text for ent in doc2.ents if ent.label_ == 'FAC']
    setting_list.append(fac_r)
    
    gpe_r = [ent.text for ent in doc2.ents if ent.label_ == 'GPE']
    setting_list.append(gpe_r)
    
    loc_r = [ent.text for ent in doc2.ents if ent.label_ == 'LOC']
    setting_list.append(loc_r)
    
    time_r = [ent.text for ent in doc2.ents if ent.label_ == 'TIME']
    setting_list.append(time_r)
    
    #추출된 항목들
    all_setting_words = sum(setting_list, [])
    
    #셋팅 추출 항목들의 총 수
    get_setting_list = len(all_setting_words)
    
    # Return all setting words
    return get_setting_list, all_setting_words


    # Intended Setting 
# 입력 : Surroundings matter a lot : 'alot', Somewhat important: 'impt', Not a big factor : 'notBigFactor'
def intendedSetting(intended_setting_input):
    if intended_setting_input == 'alot':
        int_setting_result = 'Surroundings matter a lot'
    elif intended_setting_input == 'impt':
        int_setting_result = 'Somewhat important'
    else: # not a big factor
        int_setting_result = 'Not a big factor'
    return int_setting_result


def Setting_analysis(text):

    essay_input_corpus = str(text) #문장입력
    essay_input_corpus = essay_input_corpus.lower()#소문자 변환
    #print('essay_input_corpus :', essay_input_corpus)
    
    sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화 > 문장으로 구분
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
    # print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
    # 리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
    filtered_setting_text = []
    for k in token_input_text:
        for j in setting_words_filter_list:
            if k == j:
                filtered_setting_text.append(j)
    
    # print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
    
    filtered_setting_text_ = set(filtered_setting_text) #중복제거
    filtered_setting_text__ = list(filtered_setting_text_) #다시 리스트로 변환
    # print (filtered_setting_text__) # 중복값 제거 확인
    
    # 셋팅의 장소관련 단어 추출
    extract_setting_words = list(find_setting_words(text))
    
    # 문장내 모든 셋팅 단어 추출
    tot_setting_words = extract_setting_words[1] + filtered_setting_text__
    
    # 셋팅단어가 포함된 문장을 찾아내서 추출하기
    # if 셋팅단어가 문장에 있다면, 그 문장을 추출(.로 split한 문장 리스트)해서 리스트로 저장한다.
    
    # print('sentences: ', sentences) # .로 구분된 전체 문장
    
    sentence_to_words = word_tokenize(essay_input_corpus) # 총 문장을 단어 리스트로 변환
    # print('sentence_to_words:', sentence_to_words)
    
    # 셋팅단어가 포함된 문장을 찾아내서 추출
    extrace_sentence_and_setting_words = [] # 이것은 "문장", '셋팅단어' ... 합쳐서 리스트로 저장
    extract_only_sentences_include_setting_words = [] # 셋팅 단어가 포함된 문장만 리스트로 저장
    for sentence in sentences: # 문장을 하나씩 꺼내온다.
        for item in tot_setting_words: # 셋팅 단어를 하나씩 꺼내온다.
            if item in word_tokenize(sentence): # 꺼낸 문장을 단어로 나누고, 그 안에 셋팅 단어가 있다면
                extrace_sentence_and_setting_words.append(sentence) # 셋팅 단어가 포함된 문장을 별도로 저장한다.
                extrace_sentence_and_setting_words.append(item) # 셋팅 단어도 추가로 저장한다. 
                
                extract_only_sentences_include_setting_words.append(sentence)
                
                
                ## 찾는 단어 수 대로 문장을 모두 별도 저장하기때문에 문장이 중복 저장된다. 한번만 문장이 저장되도록 하자. 
                ## 문장. '단어' , '단어' 이런 식으로다가 수정해야함. 중복리스트를 제거하면 됨.
    # 중복리스트를 제거한다.
    extrace_sentence_with_setting_words_re = set(extrace_sentence_and_setting_words)
    #print('extrace_sentence_and_setting_words(문장+단어)) :', extrace_sentence_with_setting_words_re)
    
    extract_only_sentences_include_setting_words_re = set(extract_only_sentences_include_setting_words) #중복제거
    #print('extract_only_sentences_include_setting_words(오직 셋팅 포함 문장):', extract_only_sentences_include_setting_words_re)
    
    # 단, 소문자로 문장이 저장되어 있어서, 동일한 원문을 찾을 수 없다. 소문자로 되어있는 문장을 통해서 대문자가 섞여있는 원문을 찾자
    # 방법) 소문자 문장을 단어로로 토크나이즈한 후 리스트로 만든다. 대문자 문장도 단어로 토크나이즈한 후 리스트로 만든다.
    # 두 개의 리스트를 비교해서 같은 단어가 3개 혹은 5개 이상 나오면 대문자 문장의 원문을 매칭한다. 끝!
    
    #아래 메소드에 리스트로된 문장 삽입, set 함수로 처리된것을 다시 list로 변환해야 첫 글자를 대문자로 바꿀 수 있다.
    lower_text_input = list(extract_only_sentences_include_setting_words_re)
    #print('lower_text_input: ', lower_text_input[0])
    
    ######################################################################################
    ###### 소문자 문장으로 대문자 포함원 원문 추출하는 함수 ########
    # essay_input_corpus : 최초의 입력문자를 스트링으로 변환한 원본
    def find_original_sentence(lower_text_input, essay_input_corpus):
        
        #1)원본 전체을 문장으로 토큰화
        sentence_tokenized = sent_tokenize(essay_input_corpus)
        #print("======================================")
        #print('sentence_tokenized:',sentence_tokenized)
        #문장으로 토큰화한 것을 리스트로 묶어서 다시 단어로 토큰화한다. 
        word_tokenized = [] #입력에세이 원본의 토큰화된 리스트화!
        for st_to in sentence_tokenized:
            word_tokenized.append(word_tokenize(st_to))
        #print("======================================")
        # 이렇게 되어 있을 것이다 -> 문장으로 구분되어 리스트로 나뉘고 다시 단어로 분할되어 리스트[['단어','단어', ...], ['단어','단어', ...]...]
        #print('word_tokenized:', word_tokenized)
        
        
        #2)다음으로 계산 추출된 소문자로 변환된 셋팅단어 포함 문장의 단어에 대해서 첫 글자를 대문자로 만든다.
        capital_text = []
        for lt in lower_text_input:  
            capital_text.append(lt.capitalize())
        #print("======================================")    
        #print('captal_text(첫글자 대문자로 변환되었는지 확인!!!!!!!!!!):', capital_text) # 잘됨!
        
        capital_token_text_list = []
        for cpt_item in capital_text:
            #단어로 분할해서 리스트에 담는다.
            capital_token_text_list.append(word_tokenize(cpt_item))
        #print("======================================")
        # 이렇게 되어 있을 것이다 -> 문장으로 구분되어 리스트로 나뉘고 다시 단어로 분할되어 리스트[['단어','단어', ...], ['단어','단어', ...]
        #print('captal_token_text_list:',capital_token_text_list)
        
        
        # 이제 아래 두개의 리스트를 비교해서 원본을 찾아야 한다.그리고 다시 찾은 원본토큰화된 단어 리스트를 문장으로 복원한다.
        
        # word_tokenized : 입력에세이 원본의 토큰화된 리스트화! (원본문장)
        # capital_token_text_list : 추출된 에세이 결과물 토큰화 (입력문장)
        
        # print('word_tokenized:', word_tokenized)
        # print(('capital_token_text_list:', capital_token_text_list))
        
        # 셋팅 표현이 포함된 최종 문장의 리트스 추출
        count_ct_item = 0
        included_character_exp = []
        for ct in capital_token_text_list:
            for wt in word_tokenized:
                for ct_item in ct:
                    if count_ct_item >= 5: # 겹치는 단어가 4개 이상이면 같은 문장이라고 판단하자 
                        # 같은 문장이기 땜누에 원본 리스트의 단어들을 하나의 문장으로 만들어서 저장하자
                        re_cpt = ' '.join(wt).capitalize()
                        included_character_exp.append(re_cpt)
                    elif ct_item in wt: # 리스트 안에 비교리스트가 있다면, 단어 수 카운트하고 for문 돌림
                        count_ct_item += 1
                        #print('count_ct_item:', count_ct_item)
                    else: # 비교 후 겹치는 값이 없다면 패스
                        pass
                    
        # 최종결과물 첫 글자 대문자로 복원
        
        # 최종 결과물 중복제거
        result_origin = set(included_character_exp) #셋팅 단어를 사용한 총 문장을 리스트로 출력
        setting_total_sentences_number = len(result_origin) # 셋팅 단어가 발견된 총 문장수를 구하라
        return result_origin, setting_total_sentences_number
    ####################################################################################
    
    
    # 셋팅 단어가 포함된 모든 문장을 추출
    find_origin_result = find_original_sentence(lower_text_input, essay_input_corpus)
    totalSettingSentences = find_origin_result[0]
    #print('totalSettingSentences:', totalSettingSentences)
    
    # 셋팅 단어가 포함된 총 문장 수
    setting_total_sentences_number_re = find_origin_result[1]
    ####################################################################################
    # 합격자들의 평균 셋팅문장 사용 수(임의로 설정, 나중에 평균값 계산해서 적용할 것)
    setting_total_sentences_number_of_admitted_student = 20
    ####################################################################################
    
    
    # 문장생성 부분  - Overall Emphasis on Setting의 첫 문장값 계산
    
    if setting_total_sentences_number_re > setting_total_sentences_number_of_admitted_student:
        less_more_numb = abs(setting_total_sentences_number_re - setting_total_sentences_number_of_admitted_student)
        over_all_sentence_1 = [less_more_numb, 'more']
    elif setting_total_sentences_number_re < setting_total_sentences_number_of_admitted_student:
        less_more_numb = abs(setting_total_sentences_number_re - setting_total_sentences_number_of_admitted_student)
        over_all_sentence_1 = [less_more_numb, 'fewer']
    elif setting_total_sentences_number_re == setting_total_sentences_number_of_admitted_student: 
        over_all_sentence_1 = ['a similar number of'] 
    else:
        pass
        
    print("#"*100)
    print(type(model))
    print("#"*100)
        
    for i in filtered_setting_text__:
        # print("#"*100)
        # print("str:",i)
        ext_setting_sim_words_key = model.wv.most_similar_cosmul(i) # 모델적용
        ######## 에러 발생함: 
        ### AttributeError: 'Word2Vec' object has no attribute 'most_similar_cosmul'
    
    setting_total_count = len(filtered_setting_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현 수
    setting_count_ = len(filtered_setting_text__) # 중복제거된 setting표현 총 수
        
    result_setting_words_ratio = round(setting_total_count/total_words * 100, 2)
    #return result_setting_words_ratio
    
    ##### Overall Emphasis on Setting : 그래프 표현 부분. #####
    # Setting Indicators 계산으로 문장 전체에 사용된 총 셋팅표현 값과 합격한 학생들의 셋팅 평균값을 비교하여 비율로 계산
    # Yours essay 부분
    # setting_total_count : Setting Indicators - Yours essay 부분으로, 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현 수
    # setting_total_sentences_number_re : 셋팅 단어가 포함된 총 문장 수 ---> 그래프 표현 부분 * PPT 14page 참고
    ###############################################################################################
    group_total_cnt  = 30 # group_total_cont # Admitted Case Avg. 부분으로 합격학생들의 셋팅단어 평균값(임의 입력, 계산해서 입력해야 함) ==> 적용완료
    group_total_setting_descriptors = 80 # Setting Descriptors 합격학생들의 셋팅 문장수 평균값 ==> 적용완료
    ###############################################################################################
    
    
    # 결과해석 (!! return 값 순서 바꾸면 안됨 !! 만약 값을 추가하려면 맨 뒤에부터 추가하도록! )
    # 0. result_setting_words_ratio : 전체 문장에서 셋팅관련 단어의 사용비율(포함비율)
    # 1. total_sentences : 총 문장 수
    # 2. total_words : 총 단어 수
    # 3. setting_total_count : # 개인 에세이 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현'단어' 수 -----> 그래프로 표현 * PPT 14page 참고
    # 4. setting_count_ : # 중복제거된 setting표현 총 수
    # 5. ext_setting_sim_words_key : 셋팅설정과 유사한 단어들 추출
    # 6. totalSettingSentences : 셋팅 단어가 포함된 모든 문장을 추출
    # 7. setting_total_sentences_number_re : 개인 에세이 셋팅 단어가 포함된 총 '문장' 수 ------> 그래프로 표현 * PPT 14page 참고
    # 8. over_all_sentence_1 : 문장생성 
    # 9. tot_setting_words : 총 문장에서 셋팅 단어 추출  ---- 웹에 표시할 부분임
    # 10. group_total_cnt : # Admitted Case Avg. 부분으로 합격학생들의 셋팅'단어' 평균값 ---> 그래프로 표현 * PPT 14page 참고
    # 11. group_total_setting_descriptors : Setting Descriptors 합격학생들의 셋팅 '문장'수 평균값 ---> 그래프로 표현 * PPT 14page 참고
    # 12. filtered_setting_text : 총 셋팅 단어모음(중복포함)
    
    return result_setting_words_ratio, total_sentences, total_words, setting_total_count, setting_count_, ext_setting_sim_words_key, totalSettingSentences, setting_total_sentences_number_re, over_all_sentence_1, tot_setting_words, group_total_cnt, group_total_setting_descriptors, filtered_setting_text




##########################################################
# 650단어에서 또는 전체 단어에서 단락별 셋팅단어 활용 수 분석
# 20% intro, 60% body1,2,3 20% conclusion
##########################################################
def paragraph_divide_ratio(text):

    essay_input_corpus = str(text) #문장입력
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
    
    ######### setting 관련 단어 추출 #########
    s_a_re = Setting_analysis(text)
    tot_setting_words = s_a_re[9]

    # 구간별 셋팅 단어가 몇개씩 포함되어 있는지 계산 method
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
    print('intor:', intro_s_num)
    body_1_s_num = set_wd_conunter_each_parts(tot_setting_words, body_1_)
    print('body1:', body_1_s_num)
    body_2_s_num = set_wd_conunter_each_parts(tot_setting_words, body_2_)
    print('body2:', body_2_s_num)
    body_3_s_num = set_wd_conunter_each_parts(tot_setting_words, body_3_)
    print('body3',body_3_s_num)
    conclusion_s_num = set_wd_conunter_each_parts(tot_setting_words, conclusion)
    print('conclusion:',conclusion_s_num)

    
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
    
    print('compare_parts_grup: ', compare_parts_grup_nums) # [7, 'intro', 11, 'body #1', 9, 'body #2', 9, 'body #3', 4, 'conclusion']
    
    #순서정렬
    compare_parts_grup_nums_and_parts_sorted = sorted(compare_parts_grup_nums_and_parts, reverse=True)
    print('compare_parts_grup_nums_and_parts(sorted)', compare_parts_grup_nums_and_parts_sorted) # [11, 9, 9, 7, 4]
    print('compare_parts_grup_nums_and_parts :',compare_parts_grup_nums_and_parts)
    
    first_result = compare_parts_grup_nums_and_parts_sorted[0]
    second_result = compare_parts_grup_nums_and_parts_sorted[1]
    
    get_first_re = compare_parts_grup_nums.index(first_result) #인덱스 위치찾기
    print('get_firtst_re:',get_first_re)
    #가장 많은 표현이 들어간 부분 추출(최종값)
    first_snt_part = compare_parts_grup_nums[get_first_re + 1]
    
    get_second_re = compare_parts_grup_nums.index(second_result)
    print('get_second_re:',get_second_re)
    second_snt_part = compare_parts_grup_nums[get_second_re + 1] # 인덱스 다음 항목이 최종값

    # 결과해석
    # df_sentences: 모든 단어를 데이터프레임으로 변환
    # tot_setting_words: : 추출한 셋팅 관련 단어 리스트로 변환
    # first_snt_part: 문단중 가장 셋팅 관련 단어가 많은 부분 -> overall emphasis on setting의 3번째 문장으로 표현
    # second_snt_part: 문잔중  셋팅 관련 단어가 두번째고 많은 부분 -> overall emphasis on setting의 3번째 문장으로 표현
    # compare_parts_grup_nums_and_parts : intro body_1 body_2 body_3 conclusion 의 개인 에세이 계산 값
    
    return df_sentences, tot_setting_words, first_snt_part, second_snt_part, compare_parts_grup_nums_and_parts




from gensim import corpora, models, similarities

#질문 7개
def get_appropriate_pmt_by_pmt_nomber_seven(input_text):
    
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
    
    quada_list = []
    
    for temp in sims : 
        
        #quada_list.append(round(float(temp[1]),3))
        
        quada_list.append([temp[0],round(float(temp[1]),3)])

    return quada_list





# Setting PPT 13p
# Emphasis on Setting

# intended_setting_by_you 입력: Surroundings matter a lot : 'alot', Somewhat important: 'impt', Not a big factor : 'notBigFactor'
# prompt_no : promt_1~7
def EmphasisOnSetting(prompt_no, input_text, intended_setting_by_you):
    #intended by you setting value
    intended_re = intendedSetting(intended_setting_by_you)
    
    ##########################################################
    ##########################################################
    # 1000명의 평균값 셋팅 벨류(임의로 설정, 나중에 평균값 계산해서 적용할 것)
    group_setting_mean_value = 30 # 적용완료!
    # 3번 문항에 대한 내용 입력부분 - 합격한 학생들의 평균값 적용(임시적용, 나중에 계산해서 입력할 것)
    group_setting_mean_value_for_prompt = 'moderate emphasis' # 'heavy emphasis', 'moderate emphasis', 'minimal emphasis' 중 1개 서택
    
    # 각 구간의 셋팅 관련 표현의 합격자 평균값(intro, body1, body2, body3, conclusion)
    group_setting_parts_mean_value = []

    # prompt별로 셋팅 단어 적용 비율 
    pmt_1 = [12, 6, 6, 6, 5]
    pmt_2 = [16, 6, 7, 7, 3]
    pmt_3 = [10, 5, 5, 4, 8]
    pmt_4 = [12, 6, 6, 7, 5]
    pmt_5 = [16, 6, 6, 5, 8]
    pmt_6 = [15, 5, 6, 5, 7]
    ##########################################################
    ##########################################################
    if prompt_no == 'prompt_1':
        group_setting_parts_mean_value = pmt_1
    
    elif prompt_no == 'prompt_2':
        group_setting_parts_mean_value = pmt_2

    elif prompt_no == 'prompt_3':
        group_setting_parts_mean_value = pmt_3

    elif prompt_no == 'prompt_4':
        group_setting_parts_mean_value = pmt_4

    elif prompt_no == 'prompt_5':
        group_setting_parts_mean_value = pmt_5

    else : #prompt_no == 'prompt_1':
        group_setting_parts_mean_value = pmt_6
    

    
    #detected setting value
    detected_setting_value_re = Setting_analysis(input_text)[4]
    print('detected_setting_value_re:', detected_setting_value_re)
    
    # 조건판단, 오차를 +-20% 주자
    if detected_setting_value_re > (group_setting_mean_value + round(group_setting_mean_value * 0.2)):
        dct_result = 'Surroundings matter a lot'
        personal_setting_mean_value_for_prompt = 'heavy emphasis' # 개인 입력값 결과
        less_more_number_re = abs(detected_setting_value_re - group_setting_mean_value) # 평균값보다 디텍팅한 값이 많을 경우 몇단어가 많은지
        less_more_re = 'more' # overall Emphasis sentence 1 번째 문장 생성 부분 
    elif detected_setting_value_re == group_setting_mean_value:
        dct_result = 'Somewhat important'
        personal_setting_mean_value_for_prompt = 'moderate emphasis' # 개인 입력값 결과
        less_more_number_re = abs(detected_setting_value_re - group_setting_mean_value) # 평균값보다 디텍팅한 값이 많을 경우 몇단어가 많은지
        less_more_re = 'a similar number of' # overall Emphasis sentence 1 번째 문장 생성 부분 
    elif detected_setting_value_re <= (group_setting_mean_value + round(group_setting_mean_value * 0.2)):
        dct_result = 'Somewhat important'
        personal_setting_mean_value_for_prompt = 'moderate emphasis' # 개인 입력값 결과
        less_more_number_re = abs(detected_setting_value_re - group_setting_mean_value) # 평균값보다 디텍팅한 값이 많을 경우 몇단어가 많은지
        less_more_re = 'a similar number of' # overall Emphasis sentence 1 번째 문장 생성 부분 
    elif detected_setting_value_re >= (group_setting_mean_value - round(group_setting_mean_value * 0.2)):
        dct_result = 'Somewhat important'
        personal_setting_mean_value_for_prompt = 'moderate emphasis' # 개인 입력값 결과
        less_more_re = 'a similar number of' # overall Emphasis sentence 1 번째 문장 생성 부분 
    else: # detected_setting_value_re < group_setting_mean_value:
        dct_result = 'Not a big factor'
        personal_setting_mean_value_for_prompt = 'minimal emphasis' # 개인 입력값 결과
        less_more_number_re = abs(detected_setting_value_re - group_setting_mean_value) # 평균값보다 디텍팅한 값이 많을 경우 몇단어가 많은지
        less_more_re = 'fewer' # overall Emphasis sentence 1 번째 문장 생성 부분 
        


    # Setting Focus by 합격에세 2021_08_02 추가코드
    pmompt_01 = {'Heavy emphasis on setting': 25, 'Moderate emphasis on setting': 63, 'Setting is not a big factor': 12}
    pmompt_02 = {'Heavy emphasis on setting': 35, 'Moderate emphasis on setting': 58, 'Setting is not a big factor': 7}
    pmompt_03 = {'Heavy emphasis on setting': 24, 'Moderate emphasis on setting': 47, 'Setting is not a big factor': 29}
    pmompt_04 = {'Heavy emphasis on setting': 31, 'Moderate emphasis on setting': 55, 'Setting is not a big factor': 14}
    pmompt_05 = {'Heavy emphasis on setting': 42, 'Moderate emphasis on setting': 55, 'Setting is not a big factor': 3}
    pmompt_06 = {'Heavy emphasis on setting': 21, 'Moderate emphasis on setting': 43, 'Setting is not a big factor': 36}


############################################################################################### 

    # *Prompt #7은 자기 에세이와 가장 가까운 prompt를 1-6번 안에서 고르라고 한걸로 기억해요 - 이것을 자동으로 계산해주기
    # 이 부분을 계산하기 위해서는, 에세이와 가장 가까운 prompt의 속성을 자동으로 1~6번에서 골라줘야함
    # 유사문장비교 기술을 이용하여 가장 관련성이 높은 prompt를 선택하게 해야함
    def selected_seven(prompt_no):
        if(prompt_no == "ques_seven") : # (수정 후) 7번의 Prompt를 선택했고, 에세이를 입력했다면,
            re_prompt_no_analysis = get_appropriate_pmt_by_pmt_nomber_seven(input_text)
            print('re_prompt_no_analysis:', re_prompt_no_analysis[0][0]) # 가장 일치율이 높은 prompt를 자동으로 찾아줌
            most_sim_prompt_no_by_selected_seven = re_prompt_no_analysis[0][0]
            if most_sim_prompt_no_by_selected_seven == 0:
                prompt_no == "ques_one"
            elif most_sim_prompt_no_by_selected_seven == 1:
                prompt_no == "ques_two"
            elif most_sim_prompt_no_by_selected_seven == 3:
                prompt_no == "ques_three"
            elif most_sim_prompt_no_by_selected_seven == 4:
                prompt_no == "ques_four"
            elif most_sim_prompt_no_by_selected_seven == 5:
                prompt_no == "ques_five"
            else: # most_sim_prompt_no_by_selected_seven == 6:
                prompt_no == "ques_six"

        print('selected_prompt no:', prompt_no)
        return prompt_no

############################################################################################### 

    # Setting Preferences by Admitted Students for 'Prompt #3'
    selected_prompt_number = []
    if prompt_no == "ques_one":
        selected_prompt_number.append("prompt #.1")
        web_result_prompt_by_selected = pmompt_01
    elif prompt_no == "ques_two":
        selected_prompt_number.append("prompt #.2")
        web_result_prompt_by_selected = pmompt_02
    elif prompt_no == "ques_three":
        selected_prompt_number.append("prompt #.3")
        web_result_prompt_by_selected = pmompt_03
    elif prompt_no == "ques_four":
        selected_prompt_number.append("prompt #.4")
        web_result_prompt_by_selected = pmompt_04
    elif prompt_no == "ques_five":
        selected_prompt_number.append("prompt #.5")
        web_result_prompt_by_selected = pmompt_05
    elif prompt_no == "ques_six":
        selected_prompt_number.append("prompt #.6")
        web_result_prompt_by_selected = pmompt_06
    elif prompt_no == "ques_seven":
        selected_prompt_number.append("prompt #.7")
        # prompt 1~6 중에서 선택한 에세이와 가장 가까운 prompt에 해당하는 질문을 자동 선택하는 코드
        web_result_prompt_by_selected = selected_seven("ques_seven")
    else:
        pass
    
    # print('selected prompt number:', selected_prompt_number)
    
    # 문장 생성 부분 시작
    # Sentence 1
    if intended_re == 'Surroundings matter a lot':
        Sentence_1 = 'You aimed to give high importance on setting in your personal statement.'
    elif intended_re == 'Somewhat important':
        Sentence_1 = 'You aimed to give moderate importance on setting in your personal statement.'
    elif intended_re == 'Not a big factor':
        Sentence_1 = 'You aimed to give low importance on setting in your personal statement.'
    else:
        pass
    
    
    # 문장 삽입 조건판단( 의도한 결과와 분석결과 비교)
    def inLineWith_DifferentFrom(intended_re, dct_result):
        if intended_re == dct_result:
            in_di_re = 'in line with' # 2번 문항 생성 부분
            compare_match_re = 'matches well' # 4번 문항 생성 부분
            conicide_re = 'coincides' # 4번 문항 생성 부분
        else:
            in_di_re = 'different from' # 2번 문항 생성 부분
            compare_match_re = 'does not match' # 4번 문항 생성 부분
            conicide_re = 'does not coincide' # 4번 문항 생성 부분
        return in_di_re, compare_match_re, conicide_re
    
    in_di_result = inLineWith_DifferentFrom(intended_re, dct_result)
        
    # Sentence 2
    if dct_result == 'Surroundings matter a lot':
        Sentence_2 = ['It seems that the significance of setting is high in your writing, which is ', in_di_result[0],'your intentions.']
    elif dct_result == 'Somewhat important':
        Sentence_2 = ['It seems that the significance of setting is moderate in your writing, which is ', in_di_result[0], 'your intentions.']
    elif dct_result == 'Not a big factor':
        Sentence_2 =- ['It seems that the significance of setting is moderate in your writing, which is ', in_di_result[0], 'your intentions.']
    else:
        pass
    
    
    # 문장 삽입 조건 판단(heavy emphasis / moderate emphasis / minimal emphasis)
    
    
    # Sentence 3  - 이 부분은 합격한 학생의 평균값을 적용하는 부분임(group_setting_mean_value_for_prompt 값 설정한것 참고할 것)
    Sentence_3 = ['In addition, the admitted students tend to choose to display a', group_setting_mean_value_for_prompt, 'on setting for this prompt.']
    
    
    # Sentence 4 
    Sentence_4 = ['It', in_di_result[1], 'with your intended direction for setting while it', in_di_result[2],'with the level of emphasis shown in your essay.']
    
    ####### Overall Emphasis on Setting ######
    
    sa_re = Setting_analysis(input_text)
    words_desp_re = sa_re[8]
    
    overall_sentence_1 = ['Compared to the accepted case average for this prompt, you have spent', less_more_number_re, less_more_re,'setting indicators and', words_desp_re,'words to describe the setting.']
    
    adding_more_using_less_person = detected_setting_value_re + sa_re[7] # 8번째 리스트값이 셋팅단어가 포함된 문장임
    # print('adding_more_using_less_person :',adding_more_using_less_person)
    ######################################################################################################
    ############### 합격자 평균값(계산해서 적용할 것, 현재값은 임의로 넣었음)
    adding_more_using_less_group = 27
    # 비교하여 overall_sentence_2 를 계산하기
    if adding_more_using_less_person > adding_more_using_less_group:
        overall_sent_2 = 'adding more'
        overall_sentence_2 = ['You may consider', overall_sent_2, 'words to describe the setting in your story.']
    elif adding_more_using_less_person < adding_more_using_less_group:
        overall_sent_2 = 'using less'
        overall_sentence_2 = ['You may consider', overall_sent_2, 'words to describe the setting in your story.']
    else: #adding_more_using_less_person =  adding_more_using_less_group:
        overall_sentence_2 = 'Overall, your setting description looks good compared with the accepted average.'
        
    first_2nd_parts = paragraph_divide_ratio(input_text) # 3, 4번째의 리스트 값이 아래 들어갈 문장님
        
    over_sentence_3 = ['Dividing up the personal statement in 5 equal parts by the word count, the accepted case average indicated that most number of setting descriptors are concentrated in the',  first_2nd_parts[2], 'and',  first_2nd_parts[3],'.']
    
    # 합격자 평규값
    # group_setting_parts_mean_value
    
    first_2nd_parts[4] #함수 계산 값중에서 5번째 리스트 값
    print('개인 구간별 계산 값: ', first_2nd_parts[4]) # [7, 11, 9, 9, 4]
    
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
                
    over_sentence_4 = comp_each_parts(first_2nd_parts[4], group_setting_parts_mean_value)



    # setting analysis 이용하여 Overall Emphasis on Setting 의 그래프 구현값 추출
    # 3. setting_total_count : # 개인 에세이 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현'단어' 수 -----> 그래프로 표현 * PPT 14page 참고
    # sa_re[3]

    # 7. setting_total_sentences_number_re : 개인 에세이 셋팅 단어가 포함된 총 '문장' 수 ------> 그래프로 표현 * PPT 14page 참고
    # sa_re[7]

    # 10. group_total_cnt : # Admitted Case Avg. 부분으로 합격학생들의 셋팅'단어' 평균값 ---> 그래프로 표현 * PPT 14page 참고
    # sa_re[10]

    # 11. group_total_setting_descriptors : Setting Descriptors 합격학생들의 셋팅 '문장'수 평균값 ---> 그래프로 표현 * PPT 14page 참고
    # sa_re[11]

    # 6. totalSettingSentences[6] : setting description 문장 추출
    # sa_re[6]

    # 9. tot_setting_words : 총 문장에서 셋팅 단어 추출  ---- 웹에 표시할 부분임
    # sa_re[9]

    # 결과해석
    # intended_re : intended setting by you
    # dct_result : detected setting value of personal essay
    # group_setting_mean_value_for_prompt : 합격한 학생들의 셋팅 평균값(임의 임력값, 계산해서 적용해야 함) --- 두 값의 비교 부분에서 저굥ㅇ
    # personal_setting_mean_value_for_prompt : 개입입력값에 대한 결과 --- 두 값의 비교 부분에 적용
    # selected_prompt_number : 선택한 프롬프트 질문
    # Sentence_1 ~ 4: 1~4번째 문장
    # sa_re[3] : # 3. setting_total_count : # 개인 에세이 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현'단어' 수 -----> 그래프로 표현 * PPT 14page 참고
    # sa_re[7] : # 7. setting_total_sentences_number_re : 개인 에세이 셋팅 단어가 포함된 총 '문장' 수 ------> 그래프로 표현 * PPT 14page 참고
    # sa_re[10] : # 10. group_total_cnt : # Admitted Case Avg. 부분으로 합격학생들의 셋팅'단어' 평균값 ---> 그래프로 표현 * PPT 14page 참고
    # sa_re[11] : # 11. group_total_setting_descriptors : Setting Descriptors 합격학생들의 셋팅 '문장'수 평균값 ---> 그래프로 표현 * PPT 14page 참고
    # sa_re[6] : 6. totalSettingSentences : setting description 문장 추출
    # sa_re[9] : 9. tot_setting_words : 총 문장에서 셋팅 단어 추출  ---- 웹에 표시할 부분임
    
    data = {

        "selected_prompt_number" : selected_prompt_number,
        
        "intended_result" : intended_re,
        "detected_result" : dct_result,
        
        # "setting_preferences_by_admitted_students_list" = models.TextField(default="") 
        
        "group_setting_mean_value_for_prompt" : group_setting_mean_value_for_prompt,
        "personal_setting_mean_value_for_prompt" : personal_setting_mean_value_for_prompt,
        
        "emphasis_all_comment_1" : Sentence_1,
        "emphasis_all_comment_2" : Sentence_2,
        "emphasis_all_comment_3" : Sentence_3,
        "emphasis_all_comment_4" : Sentence_4,
        

        
        "overall_emphasis_indicators_you" : sa_re[3],
        "overall_emphasis_indicators_admitted" : sa_re[7],
        "overall_emphasis_descriptors_you" : sa_re[10],
        "overall_emphasis_descriptors_admitted" : sa_re[11],
        
        # "emphasis_admitted_case_list" : "0,0,0,0,0",
        # "emphasis_your_essay_list" : "0,0,0,0,0",
        
        "emphasis_your_comment_1" : overall_sentence_1,
        "emphasis_your_comment_2" : overall_sentence_2,
        "emphasis_your_comment_3" : over_sentence_3,
        "emphasis_your_comment_4" : over_sentence_4,
        
        "indicator_descriptors" : sa_re[6],

        # 총 셋팅을 표현하는 단어 모음(시간, 장소, 공간, 전치사)
        "indicator_place_nouns" : sa_re[9],

        "Number of Setting Indicators" : sa_re[3],

        # 총 셋팅을 표현하는 전치사 모음(중복포함) ----> 이것은 전치사이기 때문에 분석에 큰 의미가 없음, DB에 적용할 필요는 없음
        "Number of total setting words(preposition)" : sa_re[12],

        # web_result_prompt_by_selected 결과로 웹사이트에 표시하는 부분,선택한 Prompt에 해당하는 합격생들의 이상적인 값 표시로 고정값
        # 예) 이하 5가지로 문항별 고정값을 웹페이지에 표시한다. 개인 에세이 분석 결과값은 위에서 계산한 값을 표시해야 함

        "web_result_prompt_by_selected" : web_result_prompt_by_selected 
        
    }

    return data 



###### run #######

# 입력

# Personal Essay Sample #1 (Prompt #1이나 Prompt #4에 맞는 아름다운 에세이)

input_text = """My hand lingered on the cold metal doorknob. I closed my eyes as the Vancouver breeze ran its chilling fingers through my hair. The man I was about to meet was infamous for demanding perfection. But the beguiling music that faintly fluttered past the unlatched window’s curtain drew me forward, inviting me to cross the threshold. Stepping into the apartment, under the watchful gaze of an emerald-eyed cat portrait, I entered the sweeping B Major scale.

Led by my intrinsic attraction towards music, coupled with the textured layers erupting the instant my fingers grazed the ivory keys, driving the hammers to shoot vibrations up in the air all around me, I soon fell in love with this new extension of my body and mind. My mom began to notice my aptitude for piano when I began returning home with trophies in my arms. These precious experiences fueled my conviction as a rising musician, but despite my confidence, I felt like something was missing.

Back in the drafty apartment, I smiled nervously and walked towards the piano from which the music emanated. Ian Parker, my new piano teacher, eyes-closed and dressed in black glided his hands effortlessly across the keys. I stood beside a leather chair, waiting as he finished the phrase. He stood up. I sat down.

Chopin Black Key Etude — a piece I knew so well I could play it eyes-closed. I took a breath and positioned my right hand in a G-flat 2nd inversion. 
Just one measure in, I was stopped. 
	“Start again.”
	Taken by surprise, I spun left. His eyes were on the score, not me. 
	I started again. Past the first measure, first phrase, then stopped again. What is going on? 
	
	“Are you listening?”
I nodded. Of course I am. 
“But are you really listening?”

As we slowly dissected each measure, I felt my confidence slip away. The piece was being chipped into fragments. Unlike my previous teachers, who listened to a full performance before giving critical feedback, Ian stopped me every five seconds. One hour later, we only got through half a page. 

Each consecutive week, the same thing happened. I struggled to meet his expectations. 
“I’m not here to teach you just how to play. I’m here to teach you how to listen.” 
I realized what Ian meant — listening involves taking what we hear and asking: is this the sound I want? What story am I telling through my interpretation? 

Absorbed in the music, I allowed my instincts and muscle memory to take over, flying past the broken tritones or neapolitan chords. But even if I was playing the right notes, it didn’t matter. Becoming immersed in the cascading arpeggio waterfalls, thundering basses, and fairydust trills was actually the easy part, which brought me joy and fueled my love for music in the first place. However, music is not just about me. True artists perform for their audience, and to bring them the same joy, to turn playing into magic-making, they must listen as the audience. 

The lesson Ian taught me echoes beyond practice rooms and concert halls. I’ve learned to listen as I explore the hidden dialogue between voices, to pauses and silence, equally as powerful as words. Listening is performing as a soloist backed up by an orchestra. Listening is calmly responding during heated debates and being the last to speak in a SPS Harkness discussion. It’s even bouncing jokes around the dining table with family. I’ve grown to envision how my voice will impact the stories of those listening to me.

To this day, my lessons with Ian continue to be tough, consisting of 80% discussion and 20% playing. When we were both so immersed in the music that I managed to get to the end of the piece before he looked up to say, “Bravo.” Now, even when I practice piano alone, I repeat my refrain: Are you listening?  """

prompt_no = 'ques_one'
intended_setting_by_you = "alot"


data = EmphasisOnSetting(prompt_no, input_text, intended_setting_by_you)
key_value_print(data)

####################################################################################################
# selected_prompt_number :  ['prompt #.1']

# intended_result :  Surroundings matter a lot
# detected_result :  Somewhat important

# group_setting_mean_value_for_prompt :  moderate emphasis
# personal_setting_mean_value_for_prompt :  moderate emphasis

# emphasis_all_comment_1 :  You aimed to give high importance on setting in your personal statement.
# emphasis_all_comment_2 :  ['It seems that the significance of setting is moderate in your writing, which is ', 'different from', 'your intentions.']
# emphasis_all_comment_3 :  ['In addition, the admitted students tend to choose to display a', 'moderate emphasis', 'on setting for this prompt.']
# emphasis_all_comment_4 :  ['It', 'does not match', 'with your intended direction for setting while it', 'does not coincide', 'with the level of emphasis shown in your essay.']

# overall_emphasis_indicators_you :  39
# overall_emphasis_indicators_admitted :  18
# overall_emphasis_descriptors_you :  70
# overall_emphasis_descriptors_admitted :  20

# emphasis_your_comment_1 :  ['Compared to the accepted case average for this prompt, you have spent', 3, 'a similar number of', 'setting indicators and', [2, 'fewer'], 'words to describe the setting.']
# emphasis_your_comment_2 :  ['You may consider', 'adding more', 'words to describe the setting in your story.']
# emphasis_your_comment_3 :  ['Dividing up the personal statement in 5 equal parts by the word count, the accepted case average indicated that most number of setting descriptors are concentrated in the', 'conclusion', 'and', 'intro', '.']
# emphasis_your_comment_4 :  ['Comparing this with your essay, we see some similarities in the pattern.']

# indicator_descriptors :  {'I heard art exploration still has plenty of openings !', 'I attempted to mimic the laid-back , grooving movements of my peers , but to no avail .', 'My movements were about as far away from ‘ jazzy ’ as one could possibly get .', 'The soft and melodic sound of a classic jazz ballad floated out of a set of speakers to my right , mixing with the confident chatter of students in the back row .', 'The band congregated in the middle of the room , each person tapping his or her feet along with the drummer .', 'Reflect on a time when you challenged a belief or idea .', 'Just relax !', 'Would you make the same decision again ?', 'As the solo section began to move around the room most students seemed relaxed and loose , acting as if they were easygoing musicians on a street corner in new orleans .', 'With my confidence already fading , i sat and thought to myself , i wonder if it ’ s too late to drop this course ?', 'I was used to the rigid accents and staccatos of the concert band world .', 'The director stepped to the front of the room , and snapped his fingers in a slow rhythm that the drummer tapped out with his worn wooden sticks .', 'As usual , i shuffled around in the back of the classroom , attempting to blend in with the sets of cubbies .', 'My shaking fingers closed around the shiny gold pieces of the saxophone in its case , leaving a streak of fingerprints down the newly cleaned exterior .', 'I felt my body tense and my eyes dart nervously around the room for a quick escape route .', 'My confidence plummeted as the person in front of me swung a closing riff and looked expectantly in my direction .', 'What prompted you to act ?', 'I scolded myself , just be jazzy and no one will notice you look out of place .'}

# indicator_place_nouns :  ['New Orleans', 'street', 'corner', 'on', 'room', 'route', 'from', 'sound', 'in', 'along', 'body', 'down', 'to']


# Number of total setting words(preposition) :  ['on', 'on', 'on', 'through', ... 'to']

# web_result_prompt_by_selected :  {'Heavy emphasis on setting': 25, 'Moderate emphasis on setting': 63, 'Setting is not a big factor': 12}
####################################################################################################

    # 결과해석
    # intended_re : intended setting by you
    # dct_result : detected setting value of personal essay
    # group_setting_mean_value_for_prompt : 합격한 학생들의 셋팅 평균값(임의 임력값, 계산해서 적용해야 함) --- 두 값의 비교 부분에서 적용
    # personal_setting_mean_value_for_prompt : 개입입력값에 대한 결과 --- 두 값의 비교 부분에 적용
    # selected_prompt_number : 선택한 프롬프트 질문
    # Sentence_1 ~ 4: 1~4번째 문장
    # sa_re[3] : # 3. setting_total_count : # 개인 에세이 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현'단어' 수 -----> 그래프로 표현 * PPT 14page 참고
    # sa_re[7] : # 7. setting_total_sentences_number_re : 개인 에세이 셋팅 단어가 포함된 총 '문장' 수 ------> 그래프로 표현 * PPT 14page 참고
    # sa_re[10] : # 10. group_total_cnt : # Admitted Case Avg. 부분으로 합격학생들의 셋팅'단어' 평균값 ---> 그래프로 표현 * PPT 14page 참고
    # sa_re[11] : # 11. group_total_setting_descriptors : Setting Descriptors 합격학생들의 셋팅 '문장'수 평균값 ---> 그래프로 표현 * PPT 14page 참고
    # sa_re[6] : 6. totalSettingSentences : setting description 문장 추출
    # sa_re[9] : 9. tot_setting_words : 총 문장에서 셋팅 단어 추출  ---- 웹에 표시할 부분임







    