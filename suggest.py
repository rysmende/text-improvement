import argparse

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def get_combinations(text: str):
    sentences = text.split('.')
    words_1 = []
    words_2 = []
    words_3 = []
    words_4 = []

    for s in sentences:
        if len(s) == 0:
            continue
        words = s.strip().split(' ')
        double_words = [
                ' '.join(words[i:i+2]) for i in range(len(words) - 1)
            ]
        triple_words = [
                ' '.join(words[i:i+3]) for i in range(len(words) - 2)
            ]
        quadle_words = [
                ' '.join(words[i:i+4]) for i in range(len(words) - 3)
            ]
        words_1.append(words)
        words_2.append(double_words)
        words_3.append(triple_words)
        words_4.append(quadle_words)

    return words_1, words_2, words_3, words_4

def get_embeddings(model, str_list):
    embeddings = model.encode(str_list)
    embeddings = torch.from_numpy(embeddings)
    return embeddings

def get_cos(word_emb, ph_emb):
    res = []
    for we in word_emb:
        cos = torch.stack([
            F.cosine_similarity(we.unsqueeze(0), ph.unsqueeze(0))
            for ph in ph_emb
        ])
        res.append(torch.stack([cos.max(), cos.argmax()]))
    return res

def suggest(model, phrases, text):
    ph_emb = get_embeddings(model, phrases)
    words_1, words_2, words_3, words_4 = get_combinations(text)
    sentence_num = len(words_1)
    for i in range(sentence_num):
        word_emb_1 = get_embeddings(model, words_1[i])
        word_emb_2 = get_embeddings(model, words_2[i])
        word_emb_3 = get_embeddings(model, words_3[i])
        word_emb_4 = get_embeddings(model, words_4[i])

        we1_ph = torch.stack(get_cos(word_emb_1, ph_emb))
        we2_ph = torch.stack(get_cos(word_emb_2, ph_emb))
        we3_ph = torch.stack(get_cos(word_emb_3, ph_emb))
        we4_ph = torch.stack(get_cos(word_emb_4, ph_emb))

        we1_max, we1_idx = we1_ph[:, 0].max(), we1_ph[:, 0].argmax()
        we2_max, we2_idx = we2_ph[:, 0].max(), we2_ph[:, 0].argmax()
        we3_max, we3_idx = we3_ph[:, 0].max(), we3_ph[:, 0].argmax()
        we4_max, we4_idx = we4_ph[:, 0].max(), we4_ph[:, 0].argmax()

        w1234 = torch.stack([we1_max, we2_max, we3_max, we4_max])
        h_score_idx = w1234.argmax()
        h_score_val = w1234.max()


        if h_score_val > 0.7:
            new_ph_len =  h_score_idx + 1

            word_idx = eval(f'we{new_ph_len}_idx').item()
            phrase_idx = eval(f'we{new_ph_len}_ph')[word_idx, 1].int().item()
            old_ph = eval(f'words_{new_ph_len}')[i][word_idx]

            new_ph = phrases[phrase_idx]
            new_ph = new_ph.lower() if word_idx else new_ph

            words_1[i][word_idx] = f'<{old_ph}> ({new_ph})'

            [words_1[i].pop(word_idx + 1) for _ in range(h_score_idx)]

        print(' '.join(words_1[i]) + '.')


with open('data/standardised_terms.txt', 'r') as f:
    phrases = [l[:-1] for l in f.readlines()]

with open('data/sample_text.txt', 'r') as f:
    text = f.read()

model = SentenceTransformer('whaleloops/phrase-bert', device='cpu')

suggest(model, phrases, text)