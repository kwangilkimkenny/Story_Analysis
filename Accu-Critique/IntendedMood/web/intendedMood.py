from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np


tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

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


def ai_emotion_analysis(input_text):
    # . 로 구분하여 리스트로 변환
    re_text = input_text.split(".")
    #print("re_text type: ", type(re_text))
        
    texts = cleaning(re_text)
    re_emot =  goemotions(texts)
    df = pd.DataFrame(re_emot)
    #print("dataframe:", df)
    label_cnt = df.count()
    #print(label_cnt)
 
    #추출된 감성중 레이블만 다시 추출하고, 이것을 리스트로 변환 후, 이중리스트 flatten하고, 가장 많이 추출된 대표감성을 카운트하여 산출한다.
    result_emotion = list(df['labels'])
    #이중리스트 flatten
    all_emo_types = sum(result_emotion, [])
    #대표감성 추출 : 리스트 항목 카운트하여 가장 높은 값 산출
    ext_emotion = {}
    for i in all_emo_types:
        if i == 'neutral': # neutral 감정은 제거함
            pass
        else:
            try: ext_emotion[i] += 1
            except: ext_emotion[i]=1    
    #print(ext_emotion)
    #결과값 오름차순 정렬 : 추출된 감성 결과가 높은 순서대로 정려하기
    key_emo = sorted(ext_emotion.items(), key=lambda x: x[1], reverse=True)
    #print("Key extract emoitons: ", key_emo)
    
    #가장 많이 추출된 감성 1개
    key_emo[0]
    #가장 많이 추출된 감성 3개
    key_emo[:2]
    result_emo_list = [*sum(zip(re_text, result_emotion),())]
    
    #결과해석
    # result_emo_list >>> 문장, 분석감성
    # key_emo[0] >>> 가장 많이 추출된 감성 1개로 이것이 에세이이 포함된 대표감성
    # key_emo[:2] 가장 많이 추출된 감성 3개
    return result_emo_list, key_emo[0], key_emo[:2]


###### Run ######
input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""

result_ = ai_emotion_analysis(input_text)

print(result_)

##### 실행하며 결과는 [문장, [감정]]으로 출력됨, 대표감성 1대, 대표감성 3개 ####

    #결과해석
    # result_emo_list >>> 문장, 분석감성
    # key_emo[0] >>> 가장 많이 추출된 감성 1개로 이것이 에세이이 포함된 대표감성
    # key_emo[:2] 가장 많이 추출된 감성 3개

# 이 결과를 토대로  html 페이지에서 동일한 문장을 찾아서 javascript로 decoration하고 
# mouseover하면 감성분석 메시지 띄우면 됨##

# ['Bloomington Normal is almost laughably cliché for a midwestern city', ['amusement'], 
# 'Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer', ['neutral']...]