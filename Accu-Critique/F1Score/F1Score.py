from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 실제 합격한 학생들의 에세이 데이터 약 100개 여기서는 30개만 
labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 
            1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1,
            1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1]	

# EssayFitAI가 분석결과로 동일한 에세이 예측결과
fitzAI = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
            1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]	

print(accuracy_score(labels, fitzAI))	# 0.9180327868852459
print(recall_score(labels, fitzAI))	# 1.0
print(precision_score(labels, fitzAI))	# 0.8958333333333334
print(f1_score(labels, fitzAI))	# 0.945054945054945


print("========= FizAI 성능 ============")
print("F1 Score: ", f1_score(labels, fitzAI))
