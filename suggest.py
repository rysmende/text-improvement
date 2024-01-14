import argparse

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-t', '--text', type=str, help='Path to text file', required=True, 
            default='data/sample_text.txt'
        )
    return parser.parse_args()

def get_combinations(text: str) -> tuple:
    '''Gets combinations of phrases in each sentence with a certain length (1-4).
    
    Keyword arguments:
        text - input text as string
    Returns:
        words_1 - unary combinations of words in each sentence
        words_2 - double combinations of words in each sentence
        words_3 - triple combinations of words in each sentence
        words_4 - quadruple combinations of words in each sentence
    '''

    # spliting the text into distinct sentences
    sentences = text.split('.')
    words_1 = []
    words_2 = []
    words_3 = []
    words_4 = []

    for s in sentences:
        # if a sentence is empty, skip
        if len(s) == 0:
            continue
        # dividing sentence into words
        words = s.strip().split(' ')
        # combining two consecutive words to form double phrases
        double_words = [
                ' '.join(words[i:i+2]) for i in range(len(words) - 1)
            ]
        # combining three consecutive words to form triple phrases
        triple_words = [
                ' '.join(words[i:i+3]) for i in range(len(words) - 2)
            ]
        # combining quadruple consecutive words to form quadruple phrases
        quadle_words = [
                ' '.join(words[i:i+4]) for i in range(len(words) - 3)
            ]
        # joining words in each sentence
        words_1.append(words)
        words_2.append(double_words)
        words_3.append(triple_words)
        words_4.append(quadle_words)

    return words_1, words_2, words_3, words_4

def get_embeddings(model, str_list: list) -> torch.Tensor:
    '''Gets embeddings from the list of strings using a pre-trained model.

    Keyword arguments:
        model - transformer model
        str_list - list of strings
    Returns:
        embeddings - batched embeddings as Tensors
    '''
    # getting embeddings
    embeddings = model.encode(str_list)
    # converting numpy embeddings as torch.Tensors
    embeddings = torch.from_numpy(embeddings)
    return embeddings

def get_cos(word_emb: torch.Tensor, ph_emb: torch.Tensor) -> list:
    '''Gets maximal cosine similarities between each pair of word combination and 
    suggested phrase embeddings. 

    Keyword arguments:
        word_emb - tensors of word embeddings
        ph_emb - tensors of phrase embeddings
    Returns:
        res - list of maximal cosine similarities between each pair
    '''
    res = []
    for we in word_emb:
        # for each word embedding getting similarity with each phrase embeddings
        cos = torch.stack([
            F.cosine_similarity(we.unsqueeze(0), ph.unsqueeze(0))
            for ph in ph_emb
        ])
        # appending a tuple of the biggest score and the index of the most suitable phrase
        res.append(torch.stack([cos.max(), cos.argmax()]))
    return res

def suggest(model, phrases: list, text: str) -> None:
    '''Prints out the initial text with phrase suggestions and scores.

    Keyword arguments:
        model - transformer model
        phrases - list of strings
        text - input string
    '''
    # getting phrase embeddings
    ph_emb = get_embeddings(model, phrases)
    # getting combinations of the input text
    words_1, words_2, words_3, words_4 = get_combinations(text)
    # getting number of sentences in the input text
    sentence_num = len(words_1)
    for i in range(sentence_num):
        # for each sentence getting word combination embeddings
        word_emb_1 = get_embeddings(model, words_1[i])
        word_emb_2 = get_embeddings(model, words_2[i])
        word_emb_3 = get_embeddings(model, words_3[i])
        word_emb_4 = get_embeddings(model, words_4[i])

        # getting cosine similarity and phrase indeces for each word combination
        we1_ph = torch.stack(get_cos(word_emb_1, ph_emb))
        we2_ph = torch.stack(get_cos(word_emb_2, ph_emb))
        we3_ph = torch.stack(get_cos(word_emb_3, ph_emb))
        we4_ph = torch.stack(get_cos(word_emb_4, ph_emb))

        # getting the maximal value of the score and the index within the word combination
        we1_max, we1_idx = we1_ph[:, 0].max(), we1_ph[:, 0].argmax()
        we2_max, we2_idx = we2_ph[:, 0].max(), we2_ph[:, 0].argmax()
        we3_max, we3_idx = we3_ph[:, 0].max(), we3_ph[:, 0].argmax()
        we4_max, we4_idx = we4_ph[:, 0].max(), we4_ph[:, 0].argmax()

        # joining the maximal score for each word combinations into a tensor 
        w1234 = torch.stack([we1_max, we2_max, we3_max, we4_max])
        # getting the index of the highest score for size of word combinations
        h_score_idx = w1234.argmax()
        # getting the maximal value of the score
        h_score_val = w1234.max()

        # if the score is greater than a certain threshold
        if h_score_val > 0.7:
            # length of the old word combination
            wc_len =  h_score_idx + 1

            # index of a word in a sentence with the highest score
            word_idx = eval(f'we{wc_len}_idx').item()
            # index of a phrase with the highest score
            phrase_idx = eval(f'we{wc_len}_ph')[word_idx, 1].int().item()
            # old word combination (old phrase)
            old_ph = eval(f'words_{wc_len}')[i][word_idx]
            # new phrase to replace old word combination
            new_ph = phrases[phrase_idx]
            # if not the first word in a sentence then lower
            new_ph = new_ph.lower() if word_idx else new_ph
            # replace old word with new word and score
            words_1[i][word_idx] = f'<{old_ph}> ({new_ph} : {h_score_val:.2f})'
            # delete consecutive words if word combination's length greater than one
            [words_1[i].pop(word_idx + 1) for _ in range(h_score_idx)]
        # print result
        print(' '.join(words_1[i]) + '.')

if __name__ == '__main__':
    args = parse_args()
    text_file = args.text

    with open('data/standardised_terms.txt', 'r') as f:
        phrases = [l[:-1] for l in f.readlines()]

    with open(text_file, 'r') as f:
        text = f.read()

    model = SentenceTransformer('whaleloops/phrase-bert', device='cpu')
    suggest(model, phrases, text)