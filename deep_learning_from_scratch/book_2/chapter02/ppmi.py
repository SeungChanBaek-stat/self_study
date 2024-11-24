import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess, create_co_matrix, most_similar, ppmi, ppmi_optimized
import time
from common import config

config.GPU = True

text = 'You say goodbye and I say hello.'

text_test = 'The sun rose over the tranquil village. Birds sang melodious tunes as gentle breezes rustled the leaves. Children laughed and played in the fields, chasing butterflies under the clear blue sky. Farmers tended to their crops, hopeful for a bountiful harvest. The baker prepared fresh bread, filling the air with a delightful aroma. Shopkeepers opened their stores, greeting neighbors with warm smiles. As the day progressed, the village buzzed with life and activity. When evening arrived, families gathered for dinner, sharing stories of their day. The stars emerged, and the village settled into peaceful slumber. Night enveloped the peaceful land, bringing rest to all.' * 100000
corpus, word_to_id, id_to_word = preprocess(text_test)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

start_time = time.time()
W = ppmi(C)
end_time = time.time()

start_time_opt = time.time()
W_opt = ppmi_optimized(C)
end_time_opt = time.time()

np.set_printoptions(precision = 3)

print('동시발생 행렬')
print(C)
print('-'*40)

print('PPMI 와 PPMI_opt 동일유무 : ', W == W_opt) # np.allclose(W, W_opt)

print('PPMI')
print(W, '\n', f'PPMI Elapsed time: {end_time - start_time}')

print('PPMI_opt')
print(W_opt, '\n', f'PPMI_opt Elapsed time: {end_time_opt - start_time_opt}')