#2020-10-11
#xgboost
#Django적용을 위한 함수화처리 완료_하지만 테스트중중중....

import pandas as pd
import numpy as np
import re
import pickle

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Tune learning_rate
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# First XGBoost model for MBTI dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##### Compute list of subject with Type | list of comments 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk import word_tokenize

import nltk
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE


#타입을 숫자로 변환

def get_types(row):
    t=row['type']

    I = 0; N = 0
    T = 0; J = 0
    
    if t[0] == 'I': I = 1
    elif t[0] == 'E': I = 0
    else: print('I-E incorrect')
        
    if t[1] == 'N': N = 1
    elif t[1] == 'S': N = 0
    else: print('N-S incorrect')
        
    if t[2] == 'T': T = 1
    elif t[2] == 'F': T = 0
    else: print('T-F incorrect')
        
    if t[3] == 'J': J = 1
    elif t[3] == 'P': J = 0
    else: print('J-P incorrect')
    return pd.Series( {'IE':I, 'NS':N , 'TF': T, 'JP': J })



#딕셔너리파일 설정
b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}

#리스트를 두개씩 묶어서 리스트로 만듬
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]



def translate_personality(personality):
    # transform mbti to binary vector
    
    return [b_Pers[l] for l in personality]


def translate_back(personality):
    # transform binary vector to mbti personality
    
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s


# We want to remove these from the psosts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
    'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

unique_type_list = [x.lower() for x in unique_type_list]

# Lemmatize
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

# Cache the stop words for speed 
cachedStopWords = stopwords.words("english")

def pre_process_data(data, remove_stop_words=True, remove_mbti_profiles=True):

    list_personality = []
    list_posts = []
    len_data = len(data)
    i=0
    
    for row in data.iterrows():
        i+=1
     

        ##### Remove and clean comments
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
            
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t,"")

        type_labelized = translate_personality(row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(temp)

    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality



def mbti_classify(text):

    # read data
    # data = pd.read_csv('/Users/jongphilkim/Desktop/Django_WEB/essayfitaiproject_2020_12_09/essayai/mbti_1.csv') 
    data = pd.read_csv('./essayai/data/mbti_1.csv') 

    # get_types 함수 적용
    data = data.join(data.apply (lambda row: get_types (row),axis=1))



    list_posts, list_personality  = pre_process_data(data, remove_stop_words=True, remove_mbti_profiles=True)

    # save
    with open('./essayai/ai_character/mbti/list_posts.pickle', 'wb') as f:
        pickle.dump(list_posts, f, pickle.HIGHEST_PROTOCOL)
        
        # save
    with open('./essayai/ai_character/mbti/list_personality.pickle', 'wb') as f:
        pickle.dump(list_personality, f, pickle.HIGHEST_PROTOCOL)
    

    # Posts to a matrix of token counts
    cntizer = CountVectorizer(analyzer="word", 
                                max_features=1500, 
                                tokenizer=None,    
                                preprocessor=None, 
                                stop_words=None,  
                                max_df=0.7,
                                min_df=0.1) 

    # Learn the vocabulary dictionary and return term-document matrix
    #print("CountVectorizer...")
    X_cnt = cntizer.fit_transform(list_posts)

    # Transform the count matrix to a normalized tf or tf-idf representation
    tfizer = TfidfTransformer()

    #print("Tf-idf...")
    # Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
    X_tfidf =  tfizer.fit_transform(X_cnt).toarray()


    feature_names = list(enumerate(cntizer.get_feature_names()))
    feature_names

    type_indicators = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) – Sensing (S)", 
                    "FT: Feeling (F) - Thinking (T)", "JP: Judging (J) – Perceiving (P)"  ]



    # Posts in tf-idf representation
    X = X_tfidf

    # setup parameters for xgboost
    param = {}
    param['n_estimators'] = 200
    param['max_depth'] = 2
    param['nthread'] = 8
    param['learning_rate'] = 0.2


    # Let's train type indicator individually
    for l in range(len(type_indicators)):
        #print("%s ..." % (type_indicators[l]))
        
        Y = list_personality[:,l]
        
        model = XGBClassifier(**param)
        
        param_grid = {
            'n_estimators' : [ 200, 300],
            'learning_rate': [ 0.2, 0.3]
            # 'learning_rate': [ 0.01, 0.1, 0.2, 0.3],
            # 'max_depth': [2,3,4],
        }
        
        
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
        grid_result = grid_search.fit(X, Y)

        # summarize results
        #print("* Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']





    my_posts = str(text)

    # The type is just a dummy so that the data prep fucntion can be reused
    mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

    my_posts, dummy  = pre_process_data(mydata, remove_stop_words=True)

    my_X_cnt = cntizer.transform(my_posts)
    my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()

    # setup parameters for xgboost
    param = {}
    param['n_estimators'] = 200
    param['max_depth'] = 2
    param['nthread'] = 8
    param['learning_rate'] = 0.2

    result = []
    # Let's train type indicator individually
    for l in range(len(type_indicators)):
        #print("%s ..." % (type_indicators[l]))
        
        Y = list_personality[:,l]

        # split data into train and test sets
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        # fit model on training data
        model = XGBClassifier(**param)
        model.fit(X_train, y_train)
        
        # make predictions for my  data
        y_pred = model.predict(my_X_tfidf)
        result.append(y_pred[0])
        # #print("* %s prediction: %s" % (type_indicators[l], y_pred))

        #print("The result is: ", translate_back(result))

        #결과를 리스트에 담고
        Result_list = list(translate_back(result))



    #mbit 결과값에 따라 내용 #print 하기
    # read data
    # data = pd.read_csv('/Users/jongphilkim/Desktop/Django_WEB/essayfitaiproject/essayai/mbti_exp.csv') 
    data = pd.read_csv('/essayai/mbti_exp.csv') 

    #새로운 데이터프레임을 만들어서 계산된 값을 추가할 예정
    df2 = pd.DataFrame(index=range(0,4),columns=['Type', 'Explain'])

    #리스트에서 한글자씩 불러와서 데이터프레임의 값을 출력하면 됨
    for i in range(0, len(Result_list)):
        type = Result_list[i]

        for j in range(0, len(data)):   
            if type == data.iloc[j,0]:
                break
                
        is_mbti = data.iloc[j,2]
        
        df2.iloc[i, [0,1]] = [type, is_mbti]    
        #print(df2)

    return df2



# A few few tweets and blog post
my_posts  = """Getting started with data science and applying machine learning has never been as simple as it is now. There are many free and paid online tutorials and courses out there to help you to get started. I’ve recently started to learn, play, and work on Data Science & Machine Learning on Kaggle.com. In this brief post, I’d like to share my experience with the Kaggle Python Docker image, which simplifies the Data Scientist’s life.
Awesome #AWS monitoring introduction.
HPE Software (now @MicroFocusSW) won the platinum reader's choice #ITAWARDS 2017 in the new category #CloudMonitoring
Certified as AWS Certified Solutions Architect 
Hi, please have a look at my Udacity interview about online learning and machine learning,
Very interesting to see the  lessons learnt during the HP Operations Orchestration to CloudSlang journey. http://bit.ly/1Xo41ci 
I came across a post on devopsdigest.com and need your input: “70% DevOps organizations Unhappy with DevOps Monitoring Tools”
In a similar investigation I found out that many DevOps organizations use several monitoring tools in parallel. Senu, Nagios, LogStach and SaaS offerings such as DataDog or SignalFX to name a few. However, one element is missing: Consolidation of alerts and status in a single pane of glass, which enables fast remediation of application and infrastructure uptime and performance issues.
Sure, there are commercial tools on the market for exactly this use case but these tools are not necessarily optimized for DevOps.
So, here my question to you: In your DevOps project, have you encountered that the lack of consolidation of alerts and status is a real issue? If yes, how did you approach the problem? Or is an ChatOps approach just right?
You will probably hear more and more about ChatOps - at conferences, DevOps meet-ups or simply from your co-worker at the coffee station. ChatOps is a term and concept coined by GitHub. It's about the conversation-driven development, automation, and operations.
Now the question is: why and how would I, as an ops-focused engineer, implement and use ChatOps in my organization? The next question then is: How to include my tools into the chat conversation?
Let’s begin by having a look at a use case. The Closed Looped Incidents Process (CLIP) can be rejuvenated with ChatOps. The work from the incident detection runs through monitoring until the resolution of issues in your application or infrastructure can be accelerated with improved, cross-team communication and collaboration.
In this blog post, I am going to describe and share my experience with deploying HP Operations Manager i 10.0 (OMi) on HP Helion Public Cloud. An Infrastructure as a Service platform such as HP Helion Public Cloud Compute is a great place to quickly spin-up a Linux server and install HP Operations Manager i for various use scenarios. An example of a good use case is monitoring workloads across public clouds such as AWS and Azure.
"""
# test = mbti_classify(my_posts)
# #print ('check') 
# test
# #print ('check2') 