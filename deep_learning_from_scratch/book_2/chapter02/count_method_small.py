import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi, ppmi_optimized

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)
W_opt = ppmi_optimized(C)

# SVD
U, S, V = np.linalg.svd(W)
U_opt, S_opt, V_opt = np.linalg.svd(W_opt)


print(C[0])  # 동시발생 행렬
print(W[0])  # PPMI 행렬
print(U[0])  # SVD

print(U[0, :2]) # SVD 결과를 2차원으로 줄임


# Plot
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha = 0.5)
plt.show()