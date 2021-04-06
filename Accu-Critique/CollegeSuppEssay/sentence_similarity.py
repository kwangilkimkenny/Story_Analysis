# 개발완료 #
# prompt, major, ps_supp_essay를 입력하면,
# College %& Department Fit, Major Fit, Prompt Oriented Setiments 값을 계산해줌

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('stsb-roberta-large')

from summarizer import Summarizer


# Sentiments analysis of prompt
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


#데이터 전처리 for Sentiment Analysis
def cleaning(datas):

    fin_datas = []
    for data in datas:
        # 영문자 이외 문자는 공백으로 변환
        only_english = re.sub('[^a-zA-Z]', ' ', data)
    
        # 데이터를 리스트에 추가 
        fin_datas.append(only_english)

    return fin_datas



# 데이터 전처리 for Sentence Summery
def cleaning_for_sent_sim(data):
    fin_data = []
    for data_itm in data:
        # 영문자 이외 문자는 공백으로 변환
        only_english = re.sub('[^a-zA-Z]', ' ', data_itm)
        lists_re_ = re.sub(r"^\s+|\s+$", "", only_english) # 공백문자 제거
        only_english_ = lists_re_.rstrip('\n')
        # 데이터를 리스트에 추가 
        fin_data.append(only_english_)
        # str 로 만드릭
        result = ' '.join(fin_data)
    return fin_data


# txt 문서 정보 불러오기 : 대학정보
def open_data_for_sent_sim(select_college):
    # 폴더 구조, 대학이름 입력 명칭을 통일해야 함
    file_path = "./college_info/college_dataset/"
    college_name = select_college
    file_name = "_college_general_info.txt"
    # file = open("./college_info/college_dataset/Brown_college_general_info.txt", 'r')
    #              ./college_info/colleges_dataset/Brown_college_general_info.txt'
    file = open(file_path + college_name + file_name, 'r')
    lists = file.readlines()
    file.close()
    
    tokenized_sentences = sent_tokenize(str(lists)) # 문장으로 토큰화
    result = ' '.join(tokenized_sentences)
    return result


# txt 문서 정보 불러오기 : college Department 정보 불러오기
def open_dept_data_for_sent_sim(select_college_dept):
    # 폴더 구조, 대학이름, department 입력 명칭을 통일해야 함
    file_path = "./college_info/dept_info_dataset/"
    college_dept_name = select_college_dept
    file_name = "_info.txt"
    # file = open("./college_info/dept_info_dataset/Brown_African Studies_dept_info.txt", 'r')
    file = open(file_path + college_dept_name + file_name, 'r')
    lists = file.readlines()
    file.close()
    
    tokenized_sentences = sent_tokenize(str(lists)) # 문장으로 토큰화
    result = ' '.join(tokenized_sentences)
    print("대학 dept 정보 불러오기 : ", result)
    return result


# txt 문서 정보 불러오기 : 선택한 전공관련 정보 추출 
# 입력값은 대학, 전공 ex) 'Browon', 'AfricanStudies'
def open_major_data_for_sent_sim(select_college, select_major):
    # 폴더 구조, 대학이름 입력 명칭을 통일해야 함
    file_path = "./major_info/major_dataset/"
    college_name = select_college
    mjr_name = select_major
    file_name = "_major_info.txt"
    # file = open("./major_info/major_dataset/Brown_AfricanStudies_major_info.txt", 'r')
    file = open(file_path + college_name + "_" + mjr_name + file_name, 'r')
    lists = file.readlines()
    file.close()
    tokenized_sentences = sent_tokenize(str(lists)) # 문장으로 토큰화
    result = ' '.join(tokenized_sentences)
    return result
   

########################################
# college 문장을 하나 가져와서, 학생의 에세이 문장들과 모두 비교 후, 평균값으로 유사도 계산 
def sent_sim_analysis(college_info_data, ps_supp_esssay_data):

    sentences = ps_supp_esssay_data
    # sentences = ["I ate dinner.", 
    #     "We had a three-course meal.", 
    #     "Brad came to dinner with us.",
    #     "He loves fish tacos.",
    #     "In the end, we all felt like we ate too much.",
    #     "We all agreed; it was a magnificent evening."]

    # Tokenization of each document
    tokenized_sent = []
    for s in sentences:
        tokenized_sent.append(word_tokenize(s.lower()))
    tokenized_sent


    def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    # from sentence_transformers import SentenceTransformer
    # sbert_model = SentenceTransformer('stsb-roberta-large')
    model = sbert_model
    sentence_embeddings = model.encode(sentences)

    # print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
    # print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])

    sim_sent_of_essay_score = 0
    counter = 0

    for q in tqdm(college_info_data):
        #query = "I had pizza and pasta"
        query = q
        print("query:", query)
        query_vec = model.encode([query])[0]

        for sent in tqdm(sentences):
            sim = cosine(query_vec, model.encode([sent])[0])
            sim_float = float(sim)
            print('sim_float:', sim_float)
            sim_sent_of_essay_score += sim_float
            counter += 1
            print("Sentence = ", sent, "; similarity = ", sim)


    re_mean = round((sim_sent_of_essay_score / counter), 2) * 100 # 일치율 추출하여 총평균 값 추출
    print("re_mean:", re_mean) # 17.0 % 의 일치율이 확률료 표시됨

    return re_mean

# # 위의 코드를 테스트용 샘플데이터 - 이하
# #### ------------start------------- ####
# # Test data
# college_info_data =  ['Founded in 1764, Brown is a leading research university home to world-renowned faculty, and also an innovative educational institution where the curiosity, creativity and intellectual joy of students drives academic excellence.n',
#  "The spirit of the undergraduate Open Curriculum infuses every aspect of the University.",
#   "Brown is a place where rigorous scholarship, complex problem-solving and service to the public good are defined by intense collaboration, intellectual discovery and working in ways that transcend traditional boundaries.", 
#   "Providence, Rhode Island — Brown's home for more than two and a half centuries — is a vibrant place to live, work and study, a stimulating hub for innovation, and a city rich in cultural diversity. Collectively, Brown's researchers are driven by the idea that their work will have a positive impact in the world.", 
#   'At the Institute at Brown for Environment and Society, scholars in the sciences, social sciences and humanities work together to confront the causes and impacts of climate change.',
#   "Brown is a leading research university, where stellar faculty and student researchers deploy deep content knowledge to generate new discoveries on those issues and many more. Nationally recognized as a leader in higher education and an accomplished scholar, she is a professor of economics and public policy.",
#   "The story of Brown is also the story of Providence and Rhode Island. Brown fosters innovative community partnerships and facilitates strong relationships with neighbors, nonprofit organizations, schools, civic organizations and businesses."]

# # Test data
# ps_supp_esssay_data = ['I inhale deeply and blow harder than I thought possible,' 'pushing the tiny ember from its resting place on the candle out into the air',
#  'My parents and the aunties and uncles around me attempt to point me in a different direction.',
#  " Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers.",
#  "At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat." ,
#  "After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water.",
#  "I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things.",
#  "I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved." ,
#  "After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution." ,
#  "While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot."]

#result_test = sent_sim_analysis(college_info_data, ps_supp_esssay_data)
#print("result :", result_test)
#### ------------ end ------------- ####
########################################

# Prompt Oriented Sentiments 
# 입력: 선택한 prompt type, 학생의 Coll Supp Essay
def PmtOrientedSentments(select_pmt_type, coll_supp_essay_input_data):
    # 학생의 Coll Supp Essay 감성 분석 결과 대표감성 5개
    ps_supp_essay_sentment_re = sentmentAnalysis_essay(coll_supp_essay_input_data)

    # sentment types by prompt
    why_us = ['admiration', 'excitement', 'pride', 'realization', 'curiosity']   
    intellectual_interest = ['curiosity', 'realization']
    meaningful_experience_n_lesson_learned = ['realization', 'approval', 'gratitude', 'admiration']
    achievement_you_are_proud_of = ['realization', 'approval', 'gratitude', 'admiration', 'pride', 'desire', 'optimism']
    social_issue_contribution_solution = ['anger', 'fear', 'disapproval', 'disappointment', 'realization', 'approval', 'gratitude', 'admiration']
    summer_activity = ['pride', 'realization', 'curiosity', 'excitement', 'amusement', 'caring']
    unique_qulity_passion_talent = ['pride', 'excitement', 'amusement', 'approval', 'admiration', 'curiosity']
    extracurricular_activity_work_experience = ['pride', 'realization', 'curiosity', 'joy', 'excitement', 'amusement', 'caring', 'optimism']
    your_community_roal_contribution = ['admiration', 'caring', 'approval', 'pride', 'gratitude', 'love']
    college_community_involvement_contirb = ['admiration', 'caring', 'approval', 'excitement', 'pride', 'gratitude']
    overcomming_challenge_ethical_dilemma = ['anger', 'fear', 'disapproval', 'disappointment', 'confusion', 'annoyed', 'realization', 'approval', 'gratitude', 'admiration', 'relief', 'optimism']
    culture_diversity = ['admiration', 'realization', 'love', 'approval', 'pride', 'gratitude']
    collaboration_teamwork = ['admiration', 'caring', 'approval', 'optimism', 'gratitude', 'love']
    creativity_projects= ['excitement', 'realization', 'curiosity', 'desire', 'amusement', 'suprise']
    leadership_experience = ['admiration', 'caring', 'approval', 'optimism', 'gratitude', 'love', 'fear', 'confusion', 'nervouseness']
    value_perspectives_beliefs = ['anger', 'fear', 'disapproval', 'disappointment', 'realization', 'approval', 'gratitude', 'admiration']
    person_who_influenced_you = ['realization', 'approval', 'gratitude', 'admiration', 'caring', 'love', 'curiosity', 'pride', 'joy']
    favorite_book_movie_quote = ['excitement', 'realization', 'curiosity', 'admiration', 'amusement', 'joy']
    write_to_future_roommate = ['admiration', 'realization', 'love', 'excitement', 'approval', 'pride', 'gratitude', 'amusement', 'curiosity', 'joy']
    diversity_inclusion_stmt = ['anger', 'fear', 'disapproval', 'disappointment', 'confusion', 'annoyed', 'realization', 'approval', 'gratitude', 'admiration','relief','optimism']
    future_goals_lesson_for_learning = ['realization', 'approval', 'gratitude', 'admiration', 'pride', 'desire', 'optimism']
    what_you_do_for_fun = ['admiration', 'excitement', 'curiosity', 'amusement','pride','joy']

    # 겹치는 감성추출값 초기화
    counter = 0
    # 겹치는 감성정보를 추출
    matching_sentment = []
    if select_pmt_type == 'Why us':
        for i in ps_supp_essay_sentment_re:
            if i in why_us:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(why_us) * 100

    elif select_pmt_type == 'Intellectual interest':
        for i in ps_supp_essay_sentment_re:
            if i in intellectual_interest:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(intellectual_interest) * 100

    elif select_pmt_type == 'Meaningful experience & lesson learned':
        for i in ps_supp_essay_sentment_re:
            if i in meaningful_experience_n_lesson_learned:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(meaningful_experience_n_lesson_learned) * 100

    elif select_pmt_type == 'Achievement you are proud of':
        for i in ps_supp_essay_sentment_re:
            if i in achievement_you_are_proud_of:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(achievement_you_are_proud_of) * 100

    elif select_pmt_type == 'Social issues: contribution & solution':
        for i in ps_supp_essay_sentment_re:
            if i in social_issue_contribution_solution:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(social_issue_contribution_solution) * 100

    elif select_pmt_type == 'Summer activity':
        for i in ps_supp_essay_sentment_re:
            if i in summer_activity:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(summer_activity) * 100

    elif select_pmt_type == 'Unique quality, passion, or talent':
        for i in ps_supp_essay_sentment_re:
            if i in unique_qulity_passion_talent:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(unique_qulity_passion_talent) * 100

    elif select_pmt_type == 'Extracurricular activity or work experience':
        for i in ps_supp_essay_sentment_re:
            if i in extracurricular_activity_work_experience:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(extracurricular_activity_work_experience) * 100

    elif select_pmt_type == 'Your community: role and contribution in your community':
        for i in ps_supp_essay_sentment_re:
            if i in your_community_roal_contribution:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(your_community_roal_contribution) * 100

    elif select_pmt_type == 'College community: intended role, involvement, and contribution in college community':
        for i in ps_supp_essay_sentment_re:
            if i in college_community_involvement_contirb:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(college_community_involvement_contirb) * 100

    elif select_pmt_type == 'Overcoming a Challenge or ethical dilemma':
        for i in ps_supp_essay_sentment_re:
            if i in overcomming_challenge_ethical_dilemma:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(overcomming_challenge_ethical_dilemma) * 100

    elif select_pmt_type == 'Culture & diversity':
        for i in ps_supp_essay_sentment_re:
            if i in culture_diversity:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(culture_diversity) * 100

    elif select_pmt_type == 'Collaboration & teamwork':
        for i in ps_supp_essay_sentment_re:
            if i in collaboration_teamwork:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(collaboration_teamwork) * 100

    elif select_pmt_type == 'Creativity/creative projects':
        for i in ps_supp_essay_sentment_re:
            if i in creativity_projects:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(creativity_projects) * 100

    elif select_pmt_type == 'Leadership experience':
        for i in ps_supp_essay_sentment_re:
            if i in leadership_experience:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(leadership_experience) * 100

    elif select_pmt_type == 'Values, perspectives, or beliefs':
        for i in ps_supp_essay_sentment_re:
            if i in value_perspectives_beliefs:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(value_perspectives_beliefs) * 100

    elif select_pmt_type == 'Person who influenced you':
        for i in ps_supp_essay_sentment_re:
            if i in person_who_influenced_you:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(person_who_influenced_you) * 100

    elif select_pmt_type == 'Favorite book/movie/quote':
        for i in ps_supp_essay_sentment_re:
            if i in favorite_book_movie_quote:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(favorite_book_movie_quote) * 100

    elif select_pmt_type == 'Write to future roommate':
        for i in ps_supp_essay_sentment_re:
            if i in write_to_future_roommate:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(write_to_future_roommate) * 100

    elif select_pmt_type == 'Diversity & Inclusion Statement':
        for i in ps_supp_essay_sentment_re:
            if i in diversity_inclusion_stmt:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(diversity_inclusion_stmt) * 100

    elif select_pmt_type == 'Future goals or reasons for learning':
        for i in ps_supp_essay_sentment_re:
            if i in future_goals_lesson_for_learning:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = counter / len(future_goals_lesson_for_learning) * 100

    else: # select_pmt_type == 'What you do for fun':
        for i in ps_supp_essay_sentment_re:
            if i in what_you_do_for_fun:
                counter += 1
                matching_sentment.append(i)
        matching_ratio = round(counter / len(what_you_do_for_fun) * 100, 2)

    # prompt oriented sentiment 최종 결과값 계산하기
    if matching_ratio >= 80:
        match_result = "Superb"
    elif matching_ratio < 80 and matching_ratio >= 60:
        match_result = "Strong"
    elif matching_ratio < 60 and matching_ratio >= 40:
        match_result = "Good"
    elif matching_ratio < 40 and matching_ratio >= 20:
        match_result = "Mediocre"
    else: # matching_ratio < 20:
        match_result = "Weak"
    
    # 결과해석
    # counter : 선택한 prompt에 해당하는 coll supp essay의 대표적 감성 5개중 일치하는 상대적인 총 개수
    # matching_sentment : 매칭되는 감성 추출값
    # matching_ratio : 매칭 비율
    # match_result : 감성비교 최종 결과 산출
    return counter, matching_sentment, matching_ratio, match_result



# 학생이 입력한 에세이의 감성 분석
def sentmentAnalysis_essay(coll_supp_essay_input_data):
    # . 로 구분하여 리스트로 변환
    re_text = coll_supp_essay_input_data.split(".")
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

    return top5Emotions

# fixed Top Comment
def fixedTopComment(select_pmt_type):
    if select_pmt_type == 'Why us':
        fixed_Top_comt = """There are two key factors to consider when writing the “why us” school & major interest essay. First, you should define yourself through your intellectual interests, intended major, role in the community, and more. Secondly, you need thorough research about the college, major, and department to show strong interest. After all, it would be best if you created the “fit” between you and the college you are applying to. Meanwhile, it would help show positive sentiments such as admiration, excitement, and curiosity towards the school of your dreams."""

    elif select_pmt_type == 'Intellectual interest':
        fixed_Top_comt = """An intellectual interest essay may deal with any topic as long as it demonstrates the writer’s knowledge, analytical thinking, and creativity. Nonetheless, experts advise that displaying the depth of knowledge in your intended major area in a curious and insightful manner could provide a more precise focal point for the reviewer. Engaging ideas can be demonstrated through a healthy level of cohesion and academically-oriented verbs, while how you connect the dots between seemingly distant ideas can show how original your thoughts are."""

    elif select_pmt_type == 'Meaningful experience & lesson learned':
        fixed_Top_comt = """For meaningful experience and lessons learned, you may write about any occasion in your life as long as it had an impact on your life. One can assume that they are looking for a unique story, your own perspective, and a lesson that presented you with a positive outlook in life."""

    elif select_pmt_type == 'Achievement you are proud of':
        fixed_Top_comt = """Writing about an achievement you are proud of entails multiple elements. You may consider including words that are closely related to a noteworthy achievement. Usually, concepts like leadership, cooperation, overcoming a hardship, triumph, and more would suit such a topic. Also, there should be sentiments that convey a sense of pride, realization, appreciation, and determination while highlighting the course of your action to reach the end result."""

    elif select_pmt_type == 'Social issues: contribution & solution':
        fixed_Top_comt = """Powerful essays about social issues involve multiple elements. Your knowledge of the given issue and activism will demonstrate social awareness. Meanwhile, you should be emotionally engaged, especially with the social problems that made you angry or disappointed. Then, your realization of the issues should be backed up by your action – to bring about changes."""

    elif select_pmt_type == 'Summer activity':
        fixed_Top_comt = """Summer is a great time to pursue your diverse interests. Your summer activities may be a popular summer program held at various colleges or an internship, or a unique project you have initiated. Regardless, there should be a sense of excitement, curiosity, and realization from these meaningful summer activities. Also, intellectual endeavors relevant to your intended major may help you stand out."""

    elif select_pmt_type == 'Unique quality, passion, or talent':
        fixed_Top_comt = """To define a unique quality, passion, or talent, you should find something about you that is not considered cliché. It should be a quality that excites you and makes you proud, while demonstrating your dedication and knowledge towards such passion would be appropriate."""

    elif select_pmt_type == 'Extracurricular activity or work experience':
        fixed_Top_comt = """For the extracurricular or work experience essay, you should select an enjoyable topic that demonstrates positive qualities about you, such as dedication, passion, leadership, contribution, and more. If you’ve chosen an intellectual engagement, establishing a major fit may help you to show your focus."""

    elif select_pmt_type == 'Your community: role and contribution in your community':
        fixed_Top_comt = """Writing about one’s community involves multiple elements because it is a broad topic. It is also a crucial topic for college admissions to evaluate an applicant’s future contribution to the campus community. Therefore, your essay should demonstrate a sense of affection and pride towards the community you are a part of while defining your role and contribution to it."""

    elif select_pmt_type == 'College community: intended role, involvement, and contribution in college community':
        fixed_Top_comt = """When writing the college community essay, there are two key factors to consider. First, you should define your interest and role in the community as of today. Then, it would be best if you had thorough research about the college’s diverse student groups, facilities, and so on. After all, you want to create the “fit” between you and the college’s community for the admissions officer to imagine your role in it. Meanwhile, it would help to show positive sentiments such as admiration, excitement, and curiosity towards the school of your dreams."""

    elif select_pmt_type == 'Overcoming a Challenge or ethical dilemma':
        fixed_Top_comt = """A challenge or dilemma in your life can be a complex mix of sentiments. Usually, it starts with negative sentiments, as this type of uncertainty can stress you out. However, since the prompt intends to see how you’ve overcome such a challenge, the essay should end on a positive note with the lessons you’ve gained from the experience."""

    elif select_pmt_type == 'Culture & diversity':
        fixed_Top_comt = """Culture and diversity may involve various elements in your life, and your understanding of such variety is essential. Meanwhile, you should be able to demonstrate how you uniquely define your own culture and diversity, ultimately, to explain how you are engaged both mentally and physically."""

    elif select_pmt_type == 'Collaboration & teamwork':
        fixed_Top_comt = """Good collaboration and teamwork essays usually demonstrate one’s appreciation towards peers and ability to solve problems. This type of essay helps colleges to gauge an applicant’s capacity to work with others. It would be a good idea to describe your actions to tackle a problem together as a team."""

    elif select_pmt_type == 'Creativity/creative projects':
        fixed_Top_comt = """Creativity is difficult to define since one can apply it to all aspects of academic fields, social interactions, or our life in general. You may choose to utilize words that are commonly associated with creative activities like inventing, designing, expressing, and so on. Often, creativity is displayed through the act of connecting-the-dots between seemingly distant topics in an unexpected way. For this prompt, AI analysis will carefully examine all of the factors align in your essay."""

    elif select_pmt_type == 'Leadership experience':
        fixed_Top_comt = """Leadership-oriented essays usually demonstrate appreciation towards peers, decisiveness, and the ability to solve problems through guidance. Often, leaders also need to work up the courage to overcome the fear and nervousness that arise from the responsibility. This type of essay helps colleges to gauge an applicant’s capacity to work with others. It would be a good idea to describe your actions to tackle a problem together as a team."""

    elif select_pmt_type == 'Values, perspectives, or beliefs':
        fixed_Top_comt = """While everyone’s values, perspectives, or beliefs may vary, it’s safe to assume that the college is looking for core values that constitute a good person. To make the essay more interesting, you may consider including the struggles and dilemmas during the process of solidifying the positive values you honor. A vivid description of the learning process and clearly stating the values you believe in would make the essay more engaging for the readers."""

    elif select_pmt_type == 'Person who influenced you':
        fixed_Top_comt = """This type of essay usually focuses on the relationship between you and the person who influenced you. Although the experience you’ve had with the other can be positive or negative, the lesson learned is likely to convey a positive influence on your set of values and viewpoint."""

    elif select_pmt_type == 'Favorite book/movie/quote':
        fixed_Top_comt = """A good essay on one’s favorite book, movie, or quote should be more than just a synopsis. Beyond your knowledge about the subject, you should convey your emotions and opinion towards the topics and main idea of the story. It will highlight your distinct taste if the book, movie, or quote you have chosen is a unique one."""

    elif select_pmt_type == 'Write to future roommate':
        fixed_Top_comt = """The ‘write to your future roommate’ essay should be a candid and casual letter form that usually starts with ‘Dear Roomie.’ Nonetheless, showing that you can be a likable roommate in writing can be mindboggling. Genuinely sharing your fond memories, unique interests, bucket list in college, and even insecurities would get your future roommate excited about meeting you. On the other hand, bragging about your achievements and credentials in this essay may work against you in the admissions process."""

    elif select_pmt_type == 'Diversity & Inclusion Statement':
        fixed_Top_comt = """Commitment to diversity and inclusion is an essential value at colleges today. Sharing your identity and the experience and lessons you’ve gained in your life is crucial to show the admissions who you are and what you believe in. Whether you identify as LGBTQIA+, ethnic minority, or religious minority, the colleges will appreciate you as diversity truly matters."""

    elif select_pmt_type == 'Future goals or reasons for learning':
        fixed_Top_comt = """Future goals or reason for learning can be explained in many forms, including your future profession, role in society, self-fulfillment, and more. You may consider including words that are closely related to a type of achievement you aim for while explaining why learning is an essential part of achieving your dream. Specifying your direction through the intended major may be an effective way to pinpoint what you wish to learn in college."""

    else: # select_pmt_type == 'What you do for fun':
        fixed_Top_comt = """What you do for fun essay is an excellent opportunity to share something about you without bragging about an award or a big leadership position. Experts say that it is better to be down-to-earth with this type of prompt and write about your favorite pastime, even if it may sound trivial."""

    return fixed_Top_comt



def sent_sim_analysis_with_bert_summarizer(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data):
    College_data = open_data_for_sent_sim(select_college) # 선택한 대학의 정보가 담긴 txt 파일을 불러오고
    #print('college_data:', College_data)
    #print('===========================================================================================')
    Mjr_data = open_major_data_for_sent_sim(select_college, select_major) # 선택한 대학과 전공의 정보를 불러와서
    #print('Major_data : ', Mjr_data)
    #print('===========================================================================================')
    Dept_data = open_dept_data_for_sent_sim(select_college_dept) # dept 정보 불러오기

    # 입력정보를 요약한다. 계산을 빠르게 하기 위해서
    model = Summarizer()

    # 대학정보 요약
    college_result = model(College_data, min_length=60)
    coll_result = ''.join(college_result)
    coll_result = re.sub(r"[\n\\]", "", coll_result) # 공백문자 제거
    coll_result = coll_result.rstrip('\n')
    coll_result = sent_tokenize(coll_result)
    print('===========================================================================================')
    print('대학정보 요약 : ', coll_result)

    # Dept 정보 요약
    college_dept_result = model(Dept_data, min_length=60)
    coll_dept_result = ''.join(college_dept_result)
    coll_dept_result = re.sub(r"[\n\\]", "", coll_dept_result) # 공백문자 제거
    coll_dept_result = coll_dept_result.rstrip('\n')
    coll_dept_result = sent_tokenize(coll_dept_result)
    print('===========================================================================================')
    print('대학 Dept 정보 요약 : ', coll_dept_result)

    # 전공정보 요약
    mjr_result = model(Mjr_data, min_length=60)
    mjr_result = ''.join(mjr_result)
    mjr_result = re.sub(r"[\n\\]", "", mjr_result) # 공백문자 제거
    mjr_result = mjr_result.rstrip('\n')
    mjr_result = sent_tokenize(mjr_result)
    print('===========================================================================================')
    print('전공정보 요약 : ', mjr_result)

    # college info + dept info fit 게산하기위해 입력값 text 합치기
    coll_dept_result = coll_result + coll_dept_result

    # majorfit 계산히기 위한 text 값 가져오기

    # 에세이 요약
    ps_supp_result = model(coll_supp_essay_input_data, min_length=60)
    ps_supp_result_ = ''.join(ps_supp_result)
    ps_supp_result_ = re.sub(r"[\n\\]", "", ps_supp_result_) # 공백문자 제거
    ps_supp_result_ = ps_supp_result_.rstrip('\n')
    ps_supp_result_ = sent_tokenize(ps_supp_result_)
    print('===========================================================================================')
    print('에세이 요약 : ', ps_supp_result_)

    # 결과 계산하기, 문장 생성은 입력 값에 따라서 선택(컬리지, 전공적합성, 감성정보)
    def fit_cal(fit_ratio, col_mjr_sentment_input):
        if fit_ratio >= 80:
            result = 'Superb'
            if col_mjr_sentment_input == coll_dept_result:
                col_n_dept_fit_sentence = "Your essay shows that you have done a great job researching the college and showing your strong interest. You demonstrate a high level of understanding of the school and department you wish to study in."
                result_2 = col_n_dept_fit_sentence
            elif col_mjr_sentment_input == mjr_result:
                mjr_fit_sentence = "Regarding your fit with the intended major, your knowledge of the discipline's intellectual concepts seems quite extensive."
                result_2 = mjr_fit_sentence
            elif col_mjr_sentment_input == sentiment_result:
                sentiment_sentence = "Your essay's vibe makes you sound very interested and excited about attending the college, which is important for the admissions officials to see."
                result_2 = sentiment_sentence
            else:
                pass

        elif fit_ratio >= 60 and fit_ratio < 80:
            result = 'Strong'
            if col_mjr_sentment_input == coll_dept_result:
                col_n_dept_fit_sentence = "Your essay shows that you have done a great job researching the college and showing your strong interest. You demonstrate a high level of understanding of the school and department you wish to study in."
                result_2 = col_n_dept_fit_sentence
            elif col_mjr_sentment_input == mjr_result:
                mjr_fit_sentence = "Regarding your fit with the intended major, your knowledge of the discipline's intellectual concepts seems quite extensive."
                result_2 = mjr_fit_sentence
            elif col_mjr_sentment_input == sentiment_result:
                sentiment_sentence = "Your essay's vibe makes you sound very interested and excited about attending the college, which is important for the admissions officials to see."
                result_2 = sentiment_sentence
            else:
                pass

        elif fit_ratio < 60 and fit_ratio >= 40:
            result = 'Good'
            if col_mjr_sentment_input == coll_dept_result:
                col_n_dept_fit_sentence = "Your essay shows that you have done a good job researching the college and showing your interest. You demonstrate a satisfactory level of understanding of the school and department you wish to study in."
                result_2 = col_n_dept_fit_sentence
            elif col_mjr_sentment_input == mjr_result:
                mjr_fit_sentence = "Regarding your fit with the intended major, your knowledge of the discipline's intellectual concepts seems good."
                result_2 = mjr_fit_sentence
            elif col_mjr_sentment_input == sentiment_result:
                sentiment_sentence = "The vibe of your essay makes you sound interested in attending the college. You may want to include more emotions of admiration and excitement for the admissions officials to see."
                result_2 = sentiment_sentence
            else:
                pass

        elif fit_ratio < 40 and fit_ratio >= 20:
            result = 'Mediocre'
            if col_mjr_sentment_input == coll_dept_result:
                col_n_dept_fit_sentence = "Your essay seems to be lacking some details about the college and may not demonstrate a strong interest. You may consider doing more research on the college and department you wish to study in."
                result_2 = col_n_dept_fit_sentence
            elif col_mjr_sentment_input == mjr_result:
                mjr_fit_sentence = "Regarding your fit with the intended major, your knowledge of the discipline's intellectual concepts seems lacking."
                result_2 = mjr_fit_sentence
            elif col_mjr_sentment_input == sentiment_result:
                sentiment_sentence = "The vibe of your essay may not demonstrate enough interest or excitement about attending the college. You may want to include more emotions of admiration and excitement for the admissions officials to see."
                result_2 = sentiment_sentence
            else:
                pass

        else: # fit_ratio < 20
            result = 'Weak'
            if col_mjr_sentment_input == coll_dept_result:
                col_n_dept_fit_sentence = "Your essay seems to be lacking some details about the college and may not demonstrate a strong interest. You may consider doing more research on the college and department you wish to study in."
                result_2 = col_n_dept_fit_sentence
            elif col_mjr_sentment_input == mjr_result:
                mjr_fit_sentence = "Regarding your fit with the intended major, your knowledge of the discipline's intellectual concepts seems lacking."
                result_2 = mjr_fit_sentence
            elif col_mjr_sentment_input == sentiment_result:
                sentiment_sentence = "The vibe of your essay may not demonstrate enough interest or excitement about attending the college. You may want to include more emotions of admiration and excitement for the admissions officials to see."
                result_2 = sentiment_sentence
            else:
                pass

        # 결과해석
        # fit_ratio : 계산한 매칭 점수(확률로 높을 수록 fit 하다는 의미임)
        # result : fit_ratio를 5가지 기준으로 계산한 최종 결과값 ex) Superb, Storng, Goodl, Mediocre, Weak 중 하나가 나옴
        # result_2 : 문장생성
        return fit_ratio, result, result_2 
        

    # 대학정보, 입력에세이의 연관성 분석
    # 이 결과는 개발중간 확인 위한 값들임
    # return coll_result, ps_supp_result_

    #### College & Department Fit ### 
    coll_dept_fit_ratio = sent_sim_analysis(coll_dept_result, ps_supp_result_)
    coll_dept_result = fit_cal(coll_dept_fit_ratio, coll_dept_result)
    # 점수 1 ---> 가중치 40%
    coll_dept_re_score = coll_dept_result[0]

    #### Major Fit ####
    mjr_fit_ratio = sent_sim_analysis(mjr_result, ps_supp_result_)
    mjr_fit_result = fit_cal(mjr_fit_ratio, mjr_result)
    # 점수 2  ---> 40%
    mjr_fit_re_score = mjr_fit_result[0]

    ### Prompt Oriented Sentiments ###
    PmtOrientedSentments_result = PmtOrientedSentments(select_pmt_type, coll_supp_essay_input_data)
    # 점수 3 ---> 가중치 20%
    pmt_sent_re_score = PmtOrientedSentments_result[2]

    # 첫 문장 생성
    TopComment = fixedTopComment(select_pmt_type)
    

    # Overall 결과 산출하기
    overall_drft_sum = coll_dept_re_score * 0.4 + mjr_fit_re_score * 0.4  + pmt_sent_re_score * 0.2
    if overall_drft_sum >= 80:
        overall_result = 'Superb'
    elif overall_drft_sum < 80 and overall_drft_sum >= 60:
        overall_result = 'Strong'
    elif overall_drft_sum < 60 and overall_drft_sum >= 40:
        overall_result = 'Good'
    elif overall_drft_sum < 40 and overall_drft_sum >= 20:
        overall_result = 'Mediocre'
    else: #overall_result < 20
        overall_result = 'Weak'

    print('overall_drft_sum :', overall_drft_sum)

    ### 결과해석 ###
    # coll_dept_result : College & Department Fit ex)Weak, 생성한 문장

    # mjr_fit_result : Major Fit ex)Weak, 생성한 문장

    # TopComment : 첫번째 Selected Prompt 에 의한 고정 문장 생성

    # PmtOrientedSentments_result : 감성분석결과
        # counter : 선택한 prompt에 해당하는 coll supp essay의 대표적 감성 5개중 일치하는 상대적인 총 개수
        # matching_sentment : 매칭되는 감성 추출값
        # matching_ratio : 매칭 비율
        # match_result : 감성비교 최종 결과 산출

    # PmtOrientedSentments_result[3] : 최종 감성 상대적 비교 결과
    # overall_drft_sum : overall sum score(계산용 값)
    # overall_reault : Overall 최종 산출값

    return coll_dept_result, mjr_fit_result, TopComment, PmtOrientedSentments_result, PmtOrientedSentments_result[3], overall_drft_sum, overall_result






# ## 실행 방법 ##

# input College Supp Essay 
essay_input = """I inhale deeply and blow harder than I thought possible, pushing the tiny ember from its resting place on the candle out into the air. The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""
# 함수 사용예   ---> def sent_sim_analysis_with_bert_summarizer(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data):

# 입력: (prompt, college name, select_college_dept, intended major, supp_essay)

re_sent_sim_analy = sent_sim_analysis_with_bert_summarizer('Why us', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)

print('Result : ', re_sent_sim_analy)


# 결과값 #

# Result :  

# Result :  

# (('Weak', 'Your essay seems to be lacking some details about the college and may not demonstrate a strong interest. You may consider doing more research on the college and department you wish to study in.'), 
# ('Weak', "Regarding your fit with the intended major, your knowledge of the discipline's intellectual concepts seems lacking."), 
# 'There are two key factors to consider when writing the “why us” school & major interest essay. First, you should define yourself through your intellectual interests, intended major, role in the community, and more. Secondly, you need thorough research about the college, major, and department to show strong interest. After all, it would be best if you created the “fit” between you and the college you are applying to. Meanwhile, it would help show positive sentiments such as admiration, excitement, and curiosity towards the school of your dreams.', 
# (3, ['excitement', 'realization', 'admiration'], 60.0, 'Strong'), 
# 'Strong')