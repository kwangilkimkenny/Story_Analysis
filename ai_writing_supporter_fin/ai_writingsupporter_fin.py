
#문장을 입력하면 추천문장 20개를보여주는 코드 
#1000개의 에세이 약 10만개의 문장을 분석해서 제시해줌
import time
import pandas as pd
from pandas import DataFrame as df
from gensim import corpora, models, similarities
import pickle

# 시간 출력 함수
def show_par(input):
    time = str(input.tm_min) + 'min ' + str(input.tm_sec) + 'sec'
    return time 

print("start!")
start_time = time.time() # 시작시간 저장

#여기서 인풋문장은 전체문장이 1~3개의 문장을 입력해야 됨. 단어들로도 가능함. 
#하지만 단어(토픽)를 추출해서 적용해보는 기능은 다른 코드로 작성할 거임
#토픽모델링으로 주제어 추출, 주제어의 조합을 이용하여 데이터베이스에서 유사한 문장 10개를 추출해볼 수 도 있음.
def aiwriter(input_text):
    
    # #데이터 1000의 문장을. 로 구분하여 리스트에 넣어놓기
    # #결과값을 토대로 문장을 가져와서 출력하기
    with open('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/essay_source.pickle', 'rb') as f:
        essay_source_data = pickle.load(f)
    # print (len(essay_source_data)) # 총 98392개의 우수한 에세이 문장
    documents = essay_source_data
    df = pd.DataFrame(documents, columns =['sentences'])

    corpus = corpora.MmCorpus('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/similerlist.mm') 
    dictionary = corpora.Dictionary.load('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/similerlist.dict') 
    lsi = models.LsiModel.load('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/model.lsi') 
    
    check_time = time.time()
    print("에세이 및 데이터 전치리, 모델 파일 모두 불러오기 WorkingTime: {}".format(show_par(time.gmtime(check_time - start_time))))
    print("----------------------------------------------------------------------------------------")


    #input answer
    doc = input_text 

    vec_bow = dictionary.doc2bow(doc.lower().split())  
    # put newly obtained document to existing dictionary object
    # 새로 얻은 문서를 기존 사전 오브젝트에 넣기
    vec_lsi = lsi[vec_bow] 
    # convert new document (henceforth, call it "query") to LSI space
    # 새로운 문서(henceforth, "query"라고 함)를 LSI 공간으로 변환
    index = similarities.MatrixSimilarity(lsi[corpus]) 
    # transform corpus to LSI space and indexize it
    # 말뭉치를 LSI 공간으로 변환하고 인덱싱
    index.save('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/similerlist.index') 
    # save index object at local directory
    # 인덱스 개체를 로컬 디렉터리에 저장
    index = similarities.MatrixSimilarity.load('/Users/kyle/Desktop/SpeedUp/Ai_writingsupporter_fin/similerlist.index')
    sims = index[vec_lsi] 
    # calculate degree of similarity of the query to existing corpus
    # 기존 말뭉치에 대한 쿼리의 유사성 정도를 계산
    check_time = time.time()
    print("입력문장에 대한 유사도 계산 WorkingTime: {}".format(show_par(time.gmtime(check_time - start_time))))
    print("----------------------------------------------------------------------------------------")    


    df2 = pd.DataFrame(sims ,columns =['similarity'])
    df3 = pd.concat([df,df2], axis=1) #문장과 유사도 계산 결과를 데이터프레임으로 합치고
    df4 =  df3.sort_values(by=['similarity'], axis=0, ascending=False)
    return df4[:20] #가장 비슷한 문장 20개를 보여줌

text_input = """Share an essay on any topic of your choice. It can be one you've already written, one that responds to a different prompt """
# 당신이 선택한 어떤 주제에 대해서도 에세이를 공유하라. 이미 쓴 글일 수도 있고, 다른 프롬프트에 응답하는 글일 수도 있다.
ai_wri_re = aiwriter(text_input)
print (ai_wri_re)

end_time = time.time() # 종료시간 저장
print("TotalWorkingTime: {}".format(show_par(time.gmtime(end_time - start_time))))
print("----------------------------------------------------------------------------------------")    


### 결과 예시
# start!
# [nltk_data] Downloading package punkt to /Users/kyle/nltk_data...
# [nltk_data]   Package punkt is already up-to-date!
# [nltk_data] Downloading package vader_lexicon to
# [nltk_data]     /Users/kyle/nltk_data...
# [nltk_data]   Package vader_lexicon is already up-to-date!
# [nltk_data] Downloading package stopwords to /Users/kyle/nltk_data...
# [nltk_data]   Package stopwords is already up-to-date!
# 필요한 파일 다운로드 WorkingTime: 0min 3sec
# ----------------------------------------------------------------------------------------
# 98392
# 에세이 및 데이터 전치리, 모델 파일 모두 불러오기 WorkingTime: 0min 3sec
# ----------------------------------------------------------------------------------------
# 입력문장에 대한 유사도 계산 WorkingTime: 0min 9sec
# ----------------------------------------------------------------------------------------
#                                                sentences  similarity
# 73229  Above is a short excerpt from one of my supple...    0.993830
# 67872   Books (I couldn't choose just one): The Names...    0.993684
# 80199                 and one essay chosen from a prompt    0.993301
# 41601               One knock, in particular, stands out    0.993299
# 72813   Almost miraculously, the sound of a C-- one o...    0.992917
# 66909  This one was for Tufts, but the Duke one was v...    0.992905
# 54454   One girl in particular, Bintou, was especiall...    0.992896
# 11739   Armed with each one of Merriam-Webster's 1557...    0.992814
# 57974                    It is one of my favorite essays    0.992614
# 24440   One of my earlier memories—I was six or so, i...    0.992498
# 81240   Substantial ones, that is, over the latest Su...    0.992495
# 5456    Shaking off her layers, a reserved grin sugge...    0.992438
# 59054   Testing them with MathWorks, Lauren was ecsta...    0.992376
# 27717   I vividly remember one Saturday afternoon in ...    0.992346
# 84064   For the weight of 100 pennies, I'm only carry...    0.992220
# 76738   I grilled one of my opponent's illogical cont...    0.992205
# 54985  Margaret McMillan Trinity One Supplement: Asse...    0.992180
# 29999        From then on I wished to be no one's burden    0.992103
# 86165   Everyone in the stadium goes wild, except one...    0.992102
# 24451  The Orthodox church is not one that preaches h...    0.992098
# TotalWorkingTime: 0min 9sec
# ----------------------------------------------------------------------------------------
