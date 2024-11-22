import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

print(C) # text의 동시발생 행렬
print(type(C))

c0 = C[word_to_id['you']] # "you"의 단어 벡터
c1 = C[word_to_id['i']] # "i"의 단어 벡터

print(cos_similarity(c0, c1)) # "you"와 "i"의 코사인 유사도


text = 'This is a pen not a pencil.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

print(C) # text의 동시발생 행렬
print(type(C))

# c0 = C[word_to_id['you']] # "you"의 단어 벡터
# c1 = C[word_to_id['i']] # "i"의 단어 벡터

# print(cos_similarity(c0, c1)) # "you"와 "i"의 코사인 유사도