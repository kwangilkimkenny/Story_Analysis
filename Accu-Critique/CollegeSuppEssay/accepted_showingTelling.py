import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F
import nltk
#cuda 메모리에 여유를 주기 위해서 잠시 딜레이를 시키자
import time
from tqdm import tqdm

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
#result_all.html에서 입력받을 text를 contents에 넣고 전처리 후 데이터프레임에 넣어줌
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


    #check whether cuda is available
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    device = torch.device("cpu")  
    #device = torch.device("cuda")
    #tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-cased')
    model.to(device)



    # for text, label in test_loader :
    #     print("text:",text)
    #     print("label:",label)


    #저장된 모델을 불러온다.
    #J:\Django\EssayFit_Django\essayfitaiproject\essayfitapp\model.pt
    time.sleep(1)
    #model = torch.load("/Users/jongphilkim/Desktop/Django_WEB/essayfitaiproject/essayai/model.pt", map_location=torch.device('cpu'))
    model = torch.load("./data/model.pt", map_location=torch.device('cpu'))
    print("model loadling~")
    model.eval()




    pred_loader = test_loader

    total_loss = 0
    total_len = 0
    total_showing__ = 0
    total_telling__ = 0

    print("check!")
    for text, label in pred_loader:
        print("text:",text)
        print("label:",label)
        encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text] #text to tokenize
        padded_list =  [e + [0] * (512-len(e)) for e in encoded_list] #padding
        sample = torch.tensor(padded_list) #torch tensor로 변환
        sample, label = sample.to(device), label.to(device) #tokenized text에 label을 넣어서 Device(gpu/cpu)에 넣기 위해 준비
        labels = torch.tensor(label) #레이블을 텐서로 변환

        outputs = model(sample,labels=labels) #모델을 통해서 샘플텍스트와 레이블 입력데이터를 출력 output에 넣음
        #시간 딜레이를 주자

        _, logits = outputs #outputs를 로짓에 넣음 이것을 softmax에 넣으면 0~1 사이로 결과가 출력됨
        
        pred = torch.argmax(F.softmax(logits), dim=1) #드디어 예측한다. argmax는 리스트(계산된 값)에서 가장 큰 값을 추출하여 pred에 넣는다. 0 ~1 사이의 값이 나올거임
        # correct = pred.eq(labels) 
        showing__ = pred.eq(1) # 예측한 결과가 1과 같으면 showing이다. 그렇지 않으면 telling
        telling__ = pred.eq(0)
        total_showing__ += showing__.sum().item()
        total_telling__ += telling__.sum().item()
        # total_correct += correct.sum().item() #그 다음에는 계산을 하면 끝!
        total_len += len(pred)
        # total_len += len(labels)



    #계산 결과를 웹페이지 result_all.html 페이지에 적용
    result_showing = round(float(total_showing__/total_len),3)*100
    result_telling = round(float(total_telling__/total_len),3)*100

    result = [result_showing, result_telling]

    return result


# Lacking ideal overboard 계산

def cal_laking_ideal_overboard(one_ps_char_desc, ideal_mean):
    min_ = int(ideal_mean-ideal_mean*0.6)
    print('min_', min_)
    max_ = int(ideal_mean+ideal_mean*0.6)
    print('max_: ', max_)
    div_ = int(((ideal_mean+ideal_mean*0.6)-(ideal_mean-ideal_mean*0.6))/3)
    print('div_:', div_)


    cal_abs = abs(ideal_mean - one_ps_char_desc) # 개인 - 단체 값의 절대값계산

    print('cal_abs 절대값 :', cal_abs)
    compare = (one_ps_char_desc + ideal_mean)/7
    print('compare :', compare)

    if one_ps_char_desc > ideal_mean: # 개인점수가 평균보다 클 경우는 overboard
        if cal_abs > compare: # 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
            print("Overboard")
            result = 2
        else: #차이가 많이 안나면
            print("Ideal")
            result = 1
            
        
    elif one_ps_char_desc < ideal_mean: # 개인점수가 평균보다 작을 경우 lacking
        if cal_abs > compare: #차이가 많이나면 # 개인점수가  평균보다 작을 경우 Lacking이고 
            print("Lacking")
            result = 0
        else: #차이가 많이 안나면
            print ("Ideal")
            result = 1
            
    else:
        print("Ideal")
        result = 1
    
    return result


#######################################################################################################
########### 실행함수!!!!!!     이것을 실행하면 모든 것이 처리됨!!!   ####################################
def ai_show_telling_analysis(input_text):

    st_result = showtell_classfy(input_text)

    out_show = st_result[0] # showing result
    print("1명의 에세이 Showing:", out_show)

    # 1000명 통계 고정값(미리계산적용한 값)
    show_ideal_mean = 50 # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1000 ea dataset mean value

    out_tell = st_result[1] # telling result
    print("1명의 에세이 Telling :", out_tell)

    # 1000명 통계 고정값(미리계산적용한 값)
    tell_idel_mean = 50 # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1000 ea dataset mean value

    re_showing_ = cal_laking_ideal_overboard(out_show, show_ideal_mean) # 각각의 값 계산 (showing)
    re_telling_ = cal_laking_ideal_overboard(out_tell, tell_idel_mean) # 각각의 값 계산 (telling)

    # data = {
    #     "showing":re_showing_,
    #     "telling":re_telling_
    # }

    return re_showing_, re_telling_


def accepted_emotion():
    emotion_ratio_score_cnt = [] # nums of character 'focus on you'


    path = "./data/accepted_data/ps_essay_evaluated.csv"
    data = pd.read_csv(path)
    #Score를 인덱스로 변환하여 데이터 찾아보기
    data.set_index('Score', inplace=True)
    for i in tqdm(data.index):
        if i is not None and i >= 4:
            get_essay = data.loc[i, 'Essay']

            input_ps_essay = get_essay
            re = ai_show_telling_analysis(str(input_ps_essay))
            result = re[0]
            emotion_ratio_score_cnt.append(result)

    #print('emotion_counter:', emotion_counter)
    e_re = [y for x in emotion_ratio_score_cnt for y in x]
    # 중복감성 추출
    emo_total_count = {}
    for i in e_re:
        try: emo_total_count[i] += 1
        except: emo_total_count[i]=1


    return emo_total_count


print('Showing Telling Result : ', accepted_emotion())

