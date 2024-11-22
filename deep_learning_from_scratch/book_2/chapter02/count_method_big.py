import sys
sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi, ppmi_optimized
from dataset import ptb
import time

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('동시발생 수 계산 ...')
C = create_co_matrix(corpus, vocab_size, window_size)

print('PPMI 계산 ...')
start_time = time.time()
W = ppmi(C, verbose = True)
print('PPMI Elapsed time :', time.time() - start_time)

print('PPMI 계산 (optimized) ...')
start_time = time.time()
W_opt = ppmi_optimized(C, verbose = True)
print('PPMI optimized Elapsed time :', time.time() - start_time)

print('SVD 계산 ...')
start_time = time.time()
try :
    # truncated SVD (빠르다!)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W_opt, n_components = wordvec_size, n_iter = 5, random_state = None)
    print('truncated SVD Elapsed time :', time.time() - start_time)
except ImportError:
    # SVD (느리다)
    U, S, V = np.linalg.svd(W_opt)
    print('SVD Elapsed time :', time.time() - start_time)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top = 5)

