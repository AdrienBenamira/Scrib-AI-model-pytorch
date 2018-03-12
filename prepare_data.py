import random
from nltk.tokenize import sent_tokenize, word_tokenize
from io import open
import torchtext.vocab as vocab
import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
MIN_LENGTH_ARTICLE = 10
MIN_LENGTH_SUMMARY = 5
MAX_LENGTH = 50
dim = 100
glove = vocab.GloVe(name='6B', dim=dim)
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


def get_vect_from_word(word):
    return glove.vectors[glove.stoi[word]]


def tokenize_article_in_words(text_article):
    sentences = [word_tokenize(t) for t in sent_tokenize(text_article)]
    words = []
    for sentence in sentences:
        words.extend(sentence)
    return words


def words_into_vect(words):
    vector = None
    for i, word in enumerate(words):
        if i == 0:
            try:
                vector = get_vect_from_word(word.lower())
            except Exception:
                vector = get_vect_from_word('unk')
        elif i == 1:
            try:
                vector = torch.stack((vector, get_vect_from_word(word.lower())), 1)
            except Exception:
                vector = torch.stack((vector, get_vect_from_word('unk')), 1)
        else:
            try:
                vector = torch.cat((vector, get_vect_from_word(word.lower())), 1)
            except Exception:
                vector = torch.cat((vector, get_vect_from_word('unk')), 1)
    return vector


def create_vocab_from_articles(A):
    word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
    word2count = {"UNK": 1}
    index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
    n_words = 4  # Count default tokens
    for i, article in enumerate(A):
        words = tokenize_article_in_words(article)
        for word in words:
            if word not in word2index:
                try:
                    get_vect_from_word(word.lower())
                    word2index[word] = n_words
                    word2count[word] = 1
                    index2word[n_words] = word
                    n_words += 1
                except Exception:
                    word2count["UNK"] += 1
            else:
                word2count[word] += 1
    return word2index, word2count, index2word


def create_ini_embedding(wordtoindex):
    sample = wordtoindex.keys()
    return words_into_vect(sample)


def pairs_and_filterpairs(articles, titles, word2index, m, n):
    pairs = []
    pairs_test = []
    compteur_train = 0
    for k, article in enumerate(articles):
        compteur = 0
        if len(article.split(' ')) >= m and len(titles[k].split(' ')) >= n:
            words = article.split(' ')
            for word in words:
                try:
                    word2index[word]
                except Exception:
                    compteur = compteur + 1
            if compteur < 3:
                if compteur_train % 9 != 0 or compteur_train % 8 != 0:
                    pairs.append([article, titles[k]])
                    compteur_train = compteur_train + 1
                else:
                    pairs_test.append([article, titles[k]])
                    compteur_train = compteur_train + 1
    return pairs, pairs_test


def indexes_from_sentence(word2index, sentence):
    ind = []
    for word in sentence.split(' '):
        try:
            ind.append(word2index[word])
        except Exception:
            ind.append(3)
    return ind + [EOS_token]


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def random_batch(batch_size, pairs, word2index):
    input_seqs = []
    target_seqs = []
    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(word2index, pair[0]))
        target_seqs.append(indexes_from_sentence(word2index, pair[1]))
    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths


filename_1 = 'drive/Scrib-AI/Giga/input.txt'
filename_2 = 'drive/Scrib-AI/Giga/task1_ref0.txt'

articles = open(filename_1, encoding='utf-8').read().strip().split('\n')
titles = open(filename_2, encoding='utf-8').read().strip().split('\n')

print("Reading lines...")
wordtoindex, wordtocount, indextoword = create_vocab_from_articles(articles)
pretrained_weight = create_ini_embedding(wordtoindex).transpose(0, 1)
print('Lexique of %d words' % len(indextoword))
print("Create pairs")
pairs_train, pairs_test = pairs_and_filterpairs(articles, titles, wordtoindex, MIN_LENGTH_SUMMARY, MIN_LENGTH_ARTICLE)
print('%d pairs' % len(pairs_train))
articles_finals = []
for i in pairs_train:
    articles_finals.append(i[0])
wti, wtc, itw = create_vocab_from_articles(articles_finals)
print('%d tokens "UNK", represent %.4f ' % (wtc["UNK"], float(wtc["UNK"]) / float(len(indextoword))))

# test : random_batch(10, pairs_train, wordtoindex)
