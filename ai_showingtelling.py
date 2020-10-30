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


#입력할 문장, 전처리가 안되어 있음
contents = "State University and I possess a common vision. I, like State University, constantly work to explore the limits of nature by exceeding expectations. Long an amateur scientist, it was this drive that brought me to the University of Texas for its Student Science Training Program in 2013. Up to that point science had been my private past time, one I had yet to explore on anyone else’s terms."

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

preprossed_contents_df = sentence_to_df(contents)


preprossed_contents_df.dropna(inplace=True)
preprossed_contents_df__ = preprossed_contents_df.sample(frac=1, random_state=999)

#전처리된 데이터를 확인(데이터프레임으로 가공됨)
print(preprossed_contents_df__)

##############################################################################################
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

#파이토치에 입력하기 위해서 로딩...
ST_test_dataset = STDataset(preprossed_contents_df__)
test_loader = DataLoader(ST_test_dataset, batch_size=1, shuffle=True, num_workers=0)

#로딩되는지 확인
ST_test_dataset.__getitem__(1)
###############################################################################################
time.sleep(10)

import torch

#check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

#device = torch.device("cuda")
#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertForSequenceClassification.from_pretrained('bert-large-cased')
model.to(device)



for text, label in test_loader :
    print("text:",text)
    print("label:",label)


#저장된 모델을 불러온다.
model = torch.load("model.pt")
print("model loadling~")
model.eval()
###############################################################################################




pred_loader = test_loader

total_loss = 0
total_len = 0
total_correct = 0
print("check!")
for text, label in pred_loader:
    print("text:",text)
    print("label:",label)
    encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text] #text to tokenize
    padded_list =  [e + [0] * (512-len(e)) for e in encoded_list] #padding
    sample = torch.tensor(padded_list) #torch tensor로 변환
    sample, label = sample.to(device), label.to(device) #tokenized text에 label을 넣어서 Device(gpu/cpu)에 넣기 위해 준비
    labels = torch.tensor(label) #레이블을 텐서로 변환
    time.sleep(14)
    outputs = model(sample,labels=labels) #모델을 통해서 샘플텍스트와 레이블 입력데이터를 출력 output에 넣음
    #시간 딜레이를 주자
    time.sleep(14)
    _, logits = outputs #outputs를 로짓에 넣음 이것을 softmax에 넣으면 0~1 사이로 결과가 출력됨
    
    pred = torch.argmax(F.softmax(logits), dim=1) #드디어 예측한다. argmax는 리스트(계산된 값)에서 가장 큰 값을 추출하여 pred에 넣는다. 0 ~1 사이의 값이 나올거임
    # correct = pred.eq(labels) # 0~1 사이의 값을 레이블값과 비교하여 같으면 맞춘것이고 아니면 틀린거다. 여기서는 같을경우에는 코랙트!!!
    # total_correct += correct.sum().item() #그 다음에는 계산을 하면 끝!
    # total_len += len(labels)


#이 데이로 predict()하여 결과를 얻는다.
print('Result: ', pred)

#우선 모든 문장을 showing telling으로 계산하고, 전체 문자에서의 비율을 계산해보자. 

#자꾸 에러가 난 이유: iloc은 loc과 반대로 라벨이 아니라 순서를 나타내는 정수 인덱스(index)만 받는다.

