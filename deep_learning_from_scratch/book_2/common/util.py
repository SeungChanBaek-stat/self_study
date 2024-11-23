# coding: utf-8
import sys
sys.path.append('..')
import os
from common.np import *


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word



def create_co_matrix(corpus : np.ndarray, vocab_size : int, window_size = 1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype = np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix



def cos_similarity(x, y, eps = 1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)



def most_similar(query, word_to_id, id_to_word, word_matrix, top = 5):
    # 1. 검색어를 꺼낸다.
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 2. 코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 3. 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        # 상위 top개까지만 출력
        count += 1
        if count >= top:
            return
        


def ppmi(C : np.ndarray, verbose = False, eps = 1e-8):
    M = np.zeros_like(C, dtype = np.float32)    # PPMI를 담을 행렬
    N = np.sum(C)   # 말뭉치에 포함된 단어의 총 개수
    S = np.sum(C, axis = 0) # 각 단어의 출현 횟수
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100) == 0:
                    print('%.1f%% 완료' % (100 * cnt / total))
    return M



def ppmi_optimized(C: np.ndarray, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)    # PPMI matrix
    N = np.sum(C)   # Total number of word pairs
    S = np.sum(C, axis=0)  # Sum over columns (word occurrence counts)
    vocab_size = C.shape[0]
    total = (vocab_size * (vocab_size + 1)) // 2  # Number of elements in the upper triangle (including diagonal)
    cnt = 0

    for i in range(vocab_size):
        for j in range(i, vocab_size):
            pmi = np.log2(C[i, j] * N / (S[i] * S[j]) + eps)
            M[i, j] = max(0, pmi)
            M[j, i] = M[i, j]  # Mirror the value to the symmetric position

            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print('%.1f%% 완료' % (100 * cnt / total))
    return M



def create_contexts_target(corpus : np.ndarray, window_size = 1):
    target = corpus[window_size : -window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)



def convert_one_hot(corpus : np.ndarray, vocab_size):
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype = np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype = np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot