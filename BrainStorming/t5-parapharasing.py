#https://predictivehacks.com/how-to-paraphrase-documents-using-transformers/

# 참고할 것!!!!!!! 검토해서 적용, 업그레이드, 테스트 할 것!
# https://github.com/hetpandya/paraphrase-datasets-pretrained-models

import nltk
import json
from collections import OrderedDict

nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from torch import relu

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

 
 
# tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  # 잘 작동, 다양성이 낮음
# model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")  

# tokenizer = AutoTokenizer.from_pretrained("t5-base")  # 요약기능 뛰어남
# model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") 


# tokenizer = AutoTokenizer.from_pretrained("t5-large")  # 품질이 떯어짐
# model = AutoModelForSeq2SeqLM.from_pretrained("t5-large") 


# tokenizer = AutoTokenizer.from_pretrained("shrishail/t5_paraphrase_msrp_paws")  # 잘 작동함
# model = AutoModelForSeq2SeqLM.from_pretrained("shrishail/t5_paraphrase_msrp_paws")

tokenizer = AutoTokenizer.from_pretrained("hetpandya/t5-small-tapaco")
model = AutoModelForSeq2SeqLM.from_pretrained("hetpandya/t5-small-tapaco")



# create a function for the paraphrase
def my_paraphrase(sentence):
   
  sentence = "paraphrase: " + sentence + " </s>"
  encoding = tokenizer.encode_plus(sentence,padding=True, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
   
  outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True, #샘플링 전략 사용
    max_length=256, # 최대 디코딩 길이는 50
    top_k=200, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
    top_p=0.1, # 누적 확률이 98%인 후보집합에서만 생성
    num_return_sequences=5) #3개의 결과를 디코딩해낸다

  output = tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
 
  return(output)


 # create a function for the paraphrase
def my_paraphrase_02(sentence):
   
  sentence = "paraphrase: " + sentence + " </s>"
  encoding = tokenizer.encode_plus(sentence,padding=True, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
   
  outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True, #샘플링 전략 사용
    max_length=256, # 최대 디코딩 길이는 50
    top_k=200, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
    top_p=0.30, # 누적 확률이 98%인 후보집합에서만 생성
    num_return_sequences=5) #3개의 결과를 디코딩해낸다

  output = tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
 
  return(output)

  # create a function for the paraphrase
def my_paraphrase_03(sentence):
   
  sentence = "paraphrase: " + sentence + " </s>"
  encoding = tokenizer.encode_plus(sentence,padding=True, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
   
  outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True, #샘플링 전략 사용
    max_length=256, # 최대 디코딩 길이는 50
    top_k=200, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
    top_p=0.60, # 누적 확률이 98%인 후보집합에서만 생성
    num_return_sequences=5) #3개의 결과를 디코딩해낸다

  output = tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
 
  return(output)


  # create a function for the paraphrase
def my_paraphrase_04(sentence):
   
  sentence = "paraphrase: " + sentence + " </s>"
  encoding = tokenizer.encode_plus(sentence,padding=True, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
   
  outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True, #샘플링 전략 사용
    max_length=256, # 최대 디코딩 길이는 50
    top_k=200, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
    top_p=0.90, # 누적 확률이 98%인 후보집합에서만 생성
    num_return_sequences=5) #3개의 결과를 디코딩해낸다

  output = tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
 
  return(output)







def StartEdit():
    print("-----------------")
    print("Enter the essay you want to paraphrase.")

    draftEssay_input = input()
    print("processing...")


    ExceptionInput(draftEssay_input)

    result_essay_ai = " ".join([my_paraphrase(sent) for sent in sent_tokenize(draftEssay_input)])

    result_essay_ai_02 = " ".join([my_paraphrase_02(sent) for sent in sent_tokenize(draftEssay_input)])

    result_essay_ai_03 = " ".join([my_paraphrase_03(sent) for sent in sent_tokenize(draftEssay_input)])

    result_essay_ai_04 = " ".join([my_paraphrase_04(sent) for sent in sent_tokenize(draftEssay_input)])

    resultData = OrderedDict()
    resultData["Result_type_01"] = result_essay_ai
    resultData["Result_type_02"] = result_essay_ai_02
    resultData["Result_type_03"] = result_essay_ai_03
    resultData["Result_type_04"] = result_essay_ai_04

    print("Result 1 : ", result_essay_ai)
    print("")
    print("Result 2 : ", result_essay_ai_02)
    print("")
    print("Result 3 : ", result_essay_ai_03)
    print("")
    print("Result 4 : ", result_essay_ai_04)
    return resultData #  selected_final_script





def ExceptionInput(input_txt):
    if (input_txt.isdigit()):
        print("An incorrect input was detected. Enter your essay and try again.")
        StartEdit()

    elif (len(input_txt) <= 20):
        print("The sentence is too short. Enter your essay and try again.")
        StartEdit()

    elif (input_txt == None):
        print("Enter the essay you want to paraphrase.")
        StartEdit()


    return input_txt




# run 
StartEdit()

# Draft essay :  My father is a martial artist and he expects a lot from me. When I was little, I wanted to be a ninja. I started training in karate when I could actually hold the training sword. Like many kids from the neighborhood, my father was a huge fan of the Teenage Mutant Ninja Turtles and I lovedHe's always on my case about practicing and being more disciplined. But I know he only wants what's best for me, and I'm grateful for his guidance.I participated in a competition. There was a black guy who is really good at kicking.  I beat the record for the fastest kick in the world. He's not a bad guy, but he's a jerk. We're in this competition and he says he wants to see a kick like mine. So I said, "I'll kickYou should have seen the other guy.I kicked him back and he started backing off.Then I started to circle him, as he was backing toward me.He was on his knees.All he had were his hands.His palms were up above his head.This was all he could do.Man, I was there so fast!I won the medal and my father began to recognize me.We're at the awards ceremony.My dad goes, 'Oh, you won. You're the only one who won.'I'm like, What?I said I won and that's all, like 'no big deal'.He said he'll

