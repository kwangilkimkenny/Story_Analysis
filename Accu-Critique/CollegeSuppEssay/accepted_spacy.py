import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
from tqdm import tqdm

def ext_noun_adv_by_spacy(input_text):
    # Create an nlp object
    doc = nlp(input_text)


    # Create list of word tokens
    token_list = []
    for token in doc:
        token_list.append(token.text)



    # Create list of word tokens after removing stopwords
    filtered_sentence =[] 

    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word) 
    #print('token_list : ', token_list)
    #print('filtered_sentence : ', filtered_sentence)
    filtered_sentence_re = " ".join(filtered_sentence)
    #print('filtered_sentence_re:', filtered_sentence_re)


    nlp_doc = nlp(filtered_sentence_re)

    # Iterate over the tokens
    # extract NOUN
    ext_noun = []
    # extract ACV
    ext_adv = []
    for token in nlp_doc:
        #print(token.text, "-->", token.pos_)
        if token.pos_ == 'NOUN':
            # Print the token and its part-of-speech tag
            #print(token.text, "-->", token.pos_)
            ext_noun.append(token.text)
        elif token.pos_ == 'ADV':
            if (token.text).isalpha(): # 문자일경우만 추출
                ext_adv.append(str(token.text))
                #print(token.text, "-->", token.pos_)
            else:
                pass
        else:
            pass


    # conut NOUN
    cnt_noun = len(ext_noun) # 명사 개수
    cnt_adv = len(ext_adv) # 형용사 개수

    # 0. cnt_noun : 명사 개수
    # 1. ext_noun : 추출된 명사들
    # 2. cnt_adv : 형용사 개수
    # 3. ext_adv : 추출된 형용사들

    return cnt_noun, ext_noun, cnt_adv, ext_adv


def get_noun_adv_from_essay():
    extract_nouns = []
    extract_advs = []
    path = "./data/accepted_data/ps_essay_evaluated.csv"
    data = pd.read_csv(path)
    #Score를 인덱스로 변환하여 데이터 찾아보기
    data.set_index('Score', inplace=True)
    for i in tqdm(data.index):
        if i is not None and i >= 4:
            get_essay = data.loc[i, 'Essay']

            input_ps_essay = get_essay
            re = ext_noun_adv_by_spacy(str(input_ps_essay))
            ext_nouns = re[1]
            ext_advs = re[3]
            extract_nouns.append(ext_nouns)
            extract_advs.append(ext_advs)

    ext_nouns_re = [y for x in extract_nouns for y in x]
    ext_advs_re = [j for q in extract_advs for j in q]
    # 중복 카운팅
    def double_check(input_list):
        total_count = {}
        for i in input_list:
            try: total_count[i] += 1
            except: total_count[i]=1
        return total_count

    # 명사중복카운트
    result_mst_nouns = double_check(ext_nouns_re)
    # 형용사중복카운트
    result_mst_advs = double_check(ext_advs_re)

    data = {
        'result_mst_nouns' :result_mst_nouns,
        'result_mst_advs' : result_mst_advs
    }

    return data



### run ###

print('ext_noun_adv_by_spacy:', get_noun_adv_from_essay())


#Users/kimkwangil/Documents/001_ESSAYFITAI/Story_Analysis-master 25/Accu-Critique/CollegeSuppEssay/accepted_spacy.py
