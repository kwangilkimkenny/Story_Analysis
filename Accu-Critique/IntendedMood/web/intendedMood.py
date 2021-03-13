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


# select prompt number to get intended mood
def intended_mood_by_prompt(promptNo):
    if promptNo == 'prompt_1':
        intended_mood = ['joy', 'pride', 'approval']
    elif promptNo == "prompt_2":
        intended_mood = ['disappointment', 'fear', 'confusion']
    elif promptNo == "prompt_3":
        intended_mood = ['curiosity', 'disapproval', 'realization']
    elif promptNo == "prompt_4":
        intended_mood = ['gratitude', 'surprise', 'admiration']
    elif promptNo == "prompt_5":
        intended_mood = ['realization', 'pride', 'admiration']
    elif promptNo == "prompt_6":
        intended_mood = ['curiosity', 'excitement', 'confusion']
    elif promptNo == "prompt_7":
        intended_mood = ['joy', 'approval','disappointment', 'fear', 
                         'confusion', 'disapproval', 'realization',
                        'gratitude', 'surprise', 'admiration', 'pride',
                        'curiosity', 'excitement', ]
    else:
        pass
    
    return intended_mood


# 에세이의 감성분석, 입력값(essay, selected prompt number)
def ai_emotion_analysis(input_text, promt_number):
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
    #key_emo[0]
    
    #가장 많이 추출된 감성 3개
    #key_emo[:2]
    
    #가장 많이 추출된 감성 5개
    key_emo[:5]
    
    result_emo_list = [*sum(zip(re_text, result_emotion),())]
    
    # 결과해석
    # result_emo_list >>> 문장, 분석감성
    # key_emo[0] >>> 가장 많이 추출된 감성 1개로 이것이 에세이이 포함된 대표감성
    # key_emo[:2] 가장 많이 추출된 감성 3개
    # key_emo[:5] 가장 많이 추출된 감성 5개
    top5Emo = key_emo[:5]
    #print('top5Emo : ', top5Emo)
    top5Emotions = [] # ['approval', 'realization', 'admiration', 'excitement', 'amusement']
    top5Emotions.append(top5Emo[0][0])
    top5Emotions.append(top5Emo[1][0])
    top5Emotions.append(top5Emo[2][0])
    top5Emotions.append(top5Emo[3][0])
    top5Emotions.append(top5Emo[4][0])
    
    # 감성추출결과 분류항목 - Intended Mood 별 연관 sentiment
    disturbed =['anger', 'annoyance', 'disapproval', 'confusion', 'disappointment', 'disgust', 'anger']
    suspenseful = ['fear', 'nervousness', 'confusion', 'surprise', 'excitement']
    sad = ['disappointment', 'embarrassment', 'grief', 'remorse', 'sadness']
    joyful = ['admiration', 'amusement', 'excitement', 'joy', 'optimism']
    calm = ['caring', 'gratitude', 'realization', 'curiosity', 'admiration', 'neutral']
    
    re_mood ='' 
    for each_emo in top5Emotions:
        if each_emo in disturbed:
            re_mood = "disturbed"
        elif each_emo in suspenseful:
            re_mood = "suspensefull"
        elif each_emo in sad:
            re_mood = "sad"
        elif each_emo in joyful:
            re_mood ="joyful"
        elif each_emo in calm:
            re_mood ="calm"
        else:
            pass
        
    #입력한 에세이에서 추출한 mood의 str을 리스트로 변환    
    detected_mood = [] #결과값으로 이것을 return할 거임
    detected_mood.append(re_mood)
    
    # intended mood, prompt에서 선택한 내용대로 관련 mood 를 추출
    get_intended_mood = intended_mood_by_prompt(promt_number) # ex) ['disappointment', 'fear', 'confusion']
    
    
    #1, 2nd Senctece 생성
    if re_mood == 'disturbed':
        sentence1 = ['You’ve intended to write the essay in a disturbed mood.']
        sentence2 = ['The AI’s analysis shows that your personal statement’s mood seems to be disturbed.']

    elif re_mood == 'suspenseful':
        sentence1 = ['You’ve intended to write the essay in a suspenseful mood.']
        sentence2 = ['The AI’s analysis shows that your personal statement’s mood seems to be suspenseful.']

    elif re_mood == 'sad':
        sentence1 = ['You’ve intended to write the essay in a sad mood.']
        sentence2 = ['The AI’s analysis shows that your personal statement’s mood seems to be sad.']

    elif re_mood == 'joyful':
        sentence1 = ['You’ve intended to write the essay in a joyful mood.']
        sentence2 = ['The AI’s analysis shows that your personal statement’s mood seems to be joyful.']
                     
    elif re_mood == 'calm':
        sentence1 = ['You’ve intended to write the essay in a calm mood.']
        sentence2 = ['The AI’s analysis shows that your personal statement’s mood seems to be calm.']

    else:
        pass

                    
    # intended mood vs. your essay mood
    intendedMoodByPmt = []
    for each_mood in get_intended_mood: # prompt에서 추출된 mood를 하나씩 가져와서 에세이에서 추출된 mood와 비교
        if each_mood in disturbed:
            intendedMoodByPmt.append(each_mood) 
        elif each_mood in suspenseful:
            intendedMoodByPmt.append(each_mood)
        elif each_mood in sad:
            intendedMoodByPmt.append(each_mood)
        elif each_mood in joyful:
            intendedMoodByPmt.append(each_mood)
        elif each_mood in calm:
            intendedMoodByPmt.append(each_mood)
            
    # 비교하여 3rd Sentece 생성 
    if intendedMoodByPmt == detected_mood: # 두 개의 mood에 해당하는 리스트의 값이 같으면
        sentence3 = """It seems that the mood portrayed in your essay is coherent with what you've intended!"""
    elif intendedMoodByPmt == ['disturbed']: # 같지 않다면 다음 항목을 각각 비교
        sentence3 = """If you wish to shift the essay’s direction towards your original intention, you may consider including more conflicts and how you’ve struggled to resolve them."""
    elif intendedMoodByPmt == ['suspenseful']:
        sentence3 = """If you wish to shift the essay’s direction towards your original intention, you may consider including more incidents, actions, and dynamic elements."""
    elif intendedMoodByPmt == ['sad']:
        sentence3 = """If you wish to shift the essay’s direction towards your original intention, you may consider including more sympathetic stories about difficult times in life."""
    elif intendedMoodByPmt == ['joy']:
        sentence3 = """If you wish to shift the essay’s direction towards your original intention, you may consider including more lighthearted life stories and the positive lessons you draw from them."""
    elif intendedMoodByPmt == ['calm']:
        sentence3 = """If you wish to shift the essay’s direction towards your original intention, you may consider including more self-reflection, intellectual topics, or observations that shaped you."""
    else:
        sentence3 = """ Try Again! """
        
    #################################################################################       
    #1000 합격한 에세이의 평균 Top 5 sentiment
    #결과는 very close / somewhat close / weak 으로 나와야함
    # 각 값은 1000명의 평균에세이값을 산출하여 적용해야함, 지금 값은 dummmy values
    prompt_1_sent_mean = [('joy', 8), ('approval', 5), ('disappointment',6),('confusion',7),('gratitude',7)] 
    prompt_2_sent_mean = [('disappointment',6),('confusion',7),('joy', 8), ('approval', 5), ('disappointment',6)]
    prompt_3_sent_mean = [('curiosity',7),('disapproval',6),('disappointment',6),('confusion',7),('gratitude',7)]
    prompt_4_sent_mean = [('gratitude',8),('surprise',6),('disappointment',6),('confusion',7),('gratitude',7)]
    prompt_5_sent_mean = [('realization',5),('admiration',4),('disappointment',6),('confusion',7),('gratitude',7)]
    prompt_6_sent_mean = [('excitement',9),('confusion',5),('disappointment',6),('confusion',7),('gratitude',7)]
    prompt_7_sent_mean = [('gratitude',7),('joy',5),('disappointment',6),('confusion',7),('gratitude',7)]
    #################################################################################
    
    if promt_number == 'prompt_1': # 1번 문항을 선택했을 경우(문항선택 'prompt_1 ~ 7')
        accepted_essay_av_value = prompt_1_sent_mean
        
    elif promt_number == 'prompt_2':
        accepted_essay_av_value = prompt_2_sent_mean
        
    elif promt_number == 'prompt_3':
        accepted_essay_av_value = prompt_3_sent_mean
        
    elif promt_number == 'prompt_4':
        accepted_essay_av_value = prompt_4_sent_mean
        
    elif promt_number == 'prompt_5':
        accepted_essay_av_value = accepted_essay_av_value = prompt_5_sent_mean
        
    elif promt_number == 'prompt_6':
        accepted_essay_av_value = prompt_6_sent_mean
        
    elif promt_number == 'prompt_7':
        accepted_essay_av_value = prompt_7_sent_mean
    else:
        pass
    
    
    # 결과해석
  
    # result_emo_list: 문장 + 감성분석결과
    # intendedMoodByPmt : intended mood 
    # detected_mood : 대표 Mood
    # sentence1,sentence2, sentence3 : intended mood vs. your mood 비교결과에 대한 문장생성 커멘트 
    
    # 대표감성 5개 추출(학생 1명거임) : key_emo[:5]
    # 합격한 한생의 prompt별 대표감성 2개(1000명 평균) : accepted_essay_av_value
    
    # In-depth Sentiment Analysis 매칭되는 결과에따라서 very close / somewhat close / weak 결정
    ps_ext_emo =[] # 개인 에세이에서 추출한 5개의 대표감성
    for itm in key_emo[:5]:
        #print(itm[0])
        ps_ext_emo.append(itm[0])
 
    print(ps_ext_emo)
    
    group_ext_emo = [] # 그룹 에세이에서 추출한 5개의 평균 대표감성 5개
    for item_2 in accepted_essay_av_value:
        group_ext_emo.append(item_2[0])
    
    print(group_ext_emo)
    
    #두 값을 비교하여 very close / somewhat close / weak 결정
    #중복요소를 추출하여 카운팅하면 두 총 리스트의 값 중에서 중복요소가 몇개 있는지 알 수 있을때 유사도를 계산할 수 있음
    count={}
    sum_emo = ps_ext_emo + group_ext_emo
    for m in sum_emo:
        try: count[m] += 1
        except: count[m] = 1
    print('중복값:', count)
    
    compare_re = []
    for value in count.values(): # 딕셔너리의 벨류 값을 하나씩 가져와서 
        if value > 1: # 1보다 큰 수는 중복된 수 이기 때문에 
            compare_re.append(value) # 중복된 수를 새로운 리스트 compare_re에 넣고
        else:
            pass
        
    sum_compare_re = sum(compare_re) 
    # 리스트의 숫자를 모두 더해서 최종 비교를 할거임,
    # 총 리스틔 수는 10개이고 중복 최대값은 5개 모두가 중복되는 10이고 최소값은 0(아무것도 중복되지 않음)   0~10까지의 수로 표현됨
    print(sum_compare_re)
    
    if sum_compare_re >= 0 and sum_compare_re <= 3:
        in_depth_sent_result = 'weak'
    elif sum_compare_re > 3 and sum_compare_re <= 7:
        in_depth_sent_result = 'somewhat close'
    elif sum_compare_re > 7 :
        in_depth_sent_result = 'very close'
        
        
    # result_emo_list: 문장 + 감성분석결과
    # intendedMoodByPmt : intended mood 
    # detected_mood : 대표 Mood
    # sentence1,sentence2, sentence3 : intended mood vs. your mood 비교결과에 대한 문장생성 커멘트
    # key_emo[:5] : 학생 한명의 에세이에서 추출한 대표감성 5개
    # accepted_essay_av_value : 1000명의 합격한 학생의 대표감서 5개
    # in_depth_sent_result : 최종 심층 분석결과

    return result_emo_list, intendedMoodByPmt, detected_mood, sentence1, sentence2, sentence3, key_emo[:5], accepted_essay_av_value, in_depth_sent_result
                    



###### Run ######

# 에세이 입력
input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""

# prompt 선택 입력
prompt_no = 'prompt_1'

result_ = ai_emotion_analysis(input_text,prompt_no)

print('중복값의 벨류값의 합 :', result_)

# 결과해석  >>>.  def ai_emotion_analysis(input_text, promt_number) 코드 실행시 아래와 같은 결과가 나옴

# result_emo_list: 문장 + 감성분석결과
# intendedMoodByPmt : intended mood 
# detected_mood : 대표 Mood
# sentence1,sentence2, sentence3 : intended mood vs. your mood 비교결과에 대한 문장생성 커멘트
# key_emo[:5] : 학생 한명의 에세이에서 추출한 대표감성 5개
# accepted_essay_av_value : 1000명의 합격한 학생의 대표감서 5개
# in_depth_sent_result : 최종 심층 분석결과