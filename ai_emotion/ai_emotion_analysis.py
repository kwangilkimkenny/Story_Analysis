from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from pprint import pprint
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np

tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-ekman")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-ekman")

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


#text = """ I whispered quietly into her ear. I had waited three years to tell her in the hopes that she would understand, but she did not. Instead, she told me I was an abomination, and at that time I was fragile, so I believed her. Feeling terrible about myself, I fell into depression and gave up on school during the seventh grade. After failing all my classes that year, I began to hate school. However, when I realized that allowing my mother\u2019s words to affect me negatively would only lead to my demise, I renewed my focus on school, my grades improved, and they have been excellent ever since. Although at first words hurt me, they ultimately taught me courage and prepared me to fight ignorance and injustice.\nMy love for the written word helped me learn courage. I vividly remember my first time walking down the dusty, opaque hallways of Newark\u2019s Public Library. I picked up a stack of tattered and neglected books that, in many ways, reminded me of myself. I read the books in silence, inspired by each word I read. Each of the characters in the novels displayed courage that I hoped to embody. Percy Jackson fought valiantly against the Titan scourge, and Esmeralda Santiago fought against the American invasion of Mac\u00fan. Reading the accounts of their courage invited me to be courageous, so I began to speak out. I told my social worker about my dire situation, and she contacted my father. This courage also translated to my studies. I had previously been shy in class, but as I felt empowered by my new found courage, I began raising my hand to participate and ask questions about things I did not understand. More than authority figures, I began to see my teachers as a source of support and encouragement. The courage to build relationships with my teachers and the power of the words we have shared helped my scholarship grow tremendously.\nMy love for the spoken word also helped me prepare to speak out against ignorance and injustice. When my newly-found courage led me to join the debate club, my love for the written word was fed, as I read about the intersections between public policy and discrimination based on race, gender, and sexual preference. Reading and understanding the information was fairly easy for me, but the oral presentation required for debates was very challenging. To improve, I practiced incessantly. I worked with the upperclassmen on the debate team and I stayed after practice with my coach three days a week for three hours each day to prepare for my debates. With their help, I learned how to deliver oral arguments with the right cadence and tone to ensure my arguments were clear and concise. Though the advice my peers and coach gave me helped me tremendously, hearing their encouragement and support meant the world to me. Because of them, I developed the confidence I needed to present oral arguments that advocate for the marginalized and disenfranchised, and my passion for fighting against ignorance and injustice was born.\nBecause of my hard work and the great support I have received, I have become a successful student who takes full advantage of every course and extracurricular opportunity. The tremendous impact that the positive words of other have had on me inspired me to become a mentor for my middle and high school debate teams, which has allowed me to share the support and encouragement that was given to me with others. Ultimately, I am grateful that the curse my mother\u2019s words placed over my life has been broken by the words in my favorite novels, in my policy debates, and from my peers, teachers, and debate coach. These positive words and the courage they taught me have thus become my weapons of choice in the fight against ignorance and injustice.\n"""
def ai_emotion_analysis(input_text)
    # . 로 구분하여 리스트로 변환
    re_text = input_text.split(".")

    texts = cleaning(re_text)
    re_emot =  goemotions(texts)
    df = pd.DataFrame(re_emot)

    #결과물을 다시 감정유형별 비율로 계산하여 새로운 데이터프레임을 만들자!
    result_emotion = df['labels'].value_counts(normalize=True, sort=True, ascending=False, dropna=True) #문장 전체에서 각 값의 상대적 비율을 게산
    print (result_emotion)
    return result_emotion