# 1) Key Literary Elements (40%):
# 그냥 PS 평가 로직 그대로 활용해서 총괄 점수

# 2) Prompt Oriented Sentiments (20%):
# Main: realization, approval, admiration, gratitude
# Sub (가산점): confusion, disappointment, caring

# 3)  Originality (Topic detection & 단어 간 vector 거리) (20%)
# Key Academic Topics 간의 vector 거리 계산 해야 해요.
# -너무 가까우면 너무 재미 없는거고
# -너무 멀면 헛소리
# -어느정도 거리가 있으면 (살짝 중간 이상 거리 ?) creative 한거고
# 이거는 잘 고민해 주셔요.
# Creative (문학적) Writing Sample 필요? (linear vs. multi-dimension)

# 4) Perspective (20%) - 나의 관점, 나의 의견 표출
# 로직: Opinion/Belief words (70%) + viewpoint adverbs (30%)  + emphasizing adjectives 
