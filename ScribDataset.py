import random as rd

import torch
import torchtext.vocab as vocab
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset

from utils_data import get_vector_article, pad_sequence


class ScribDataset(Dataset):
    """
    Datasets Loader
        # Two methods to get a batch
        # The first getting a list of 50 examples if the start parameter isn't given, returns a random batch
        batch = cnn_dataset.get_batch(batch_size=50)
        # The second one: getting an iterator
        batch_gene = cnn_dataset.get_batch(batch_size=50, generator=True)
        for article in batch_gene:
            print(article)  # With this method, the article is fetched only when used

        # Get a dataloader iterator
        # If generator is set to True, the batch will be an iterator
        dataload = cnn_dataset.data_loader(batch_size=50, generator=False)
        for batch in dataload:
            print(batch)
    """

    def __init__(self, scribAPI, word_to_index, index_to_word, name="cnn-daily", transform=None, pad_token='<pad>'):
        self.pad_token = pad_token
        self.index_to_word = index_to_word
        self.word_to_index = word_to_index
        self.scribAPI = scribAPI
        self.dataset_name = name
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        self.length = self.get_length()

    def get_length(self):
        return self.scribAPI.get_dataset_length(self.dataset_name)[0]['count']

    def __len__(self):
        return self.length

    def get_articles(self, start, stop):
        articles = self.scribAPI.get_dataset(self.dataset_name, offset=start, limit=stop-start)
        if not articles:
            raise IndexError('Index out of range')
        for article in articles:
            articles_result = {}
            article.pop('ArticleDataset')
            summaries = article.pop('Summaries', [])
            articles_result['original'] = {}
            articles_result['original']['article'] = article['fullText'] if 'fullText' in article.keys() else ''
            articles_result['original']['summaries'] = [summary['content'] for summary in summaries]
            articles_result['summaries'] = [self.transform(summary['content']) for summary in summaries]
            articles_result['article'] = self.transform(article['fullText']) if 'fullText' in article.keys() else ''
            yield articles_result

    def __getitem__(self, index):
        if type(index) == slice:
            start = index.start
            stop = index.stop
        else:
            start = index
            stop = index + 1
        articles = self.get_articles(start, stop)
        result_articles = dict(summaries=[], article=[], original=[])
        for article in articles:
            result_articles['summaries'].append(article['summaries'])
            result_articles['article'].append(article['article'])
            result_articles['original'].append(article['original'])
        return result_articles

    def get_batch(self, batch_size, start=None, generator=False):
        """
        Get a batch from the database
        :param batch_size:
        :param start: id of the first article (if None, returns a random batch)
        :param generator: if True, returns a generator
        :rtype: a generator or a list of the batch
        """
        if start is None:
            start = rd.randint(0, self.length - batch_size - 1)
        if start > self.length - batch_size - 1:
            raise KeyError('Index out of range. The start key must be <= len(dataset) - batch_size - 1.')
        if generator:
            return self.get_articles(start, start + batch_size)
        articles_data = self.__getitem__(slice(start, start + batch_size))['article']
        length = [len(article) for article in articles_data]
        articles = pad_sequence(articles_data, length, self.word_to_index, self.pad_token)
        return articles

    def data_loader(self, batch_size=1, generator=False):
        """
        Get a dataloader
        :param batch_size: size of the batch (default: 1)
        :param generator: if True, returns an generator
        :rtype: an iterator
        """
        yield self.get_batch(batch_size, generator=generator)

    def append(self, article):
        """
        Add an article to the dataset
        :param article: the data to add
        :type article: dict(article="", summaries=["", ...])
        """
        self.scribAPI.add_article(self.dataset_name, article['article'], article['summaries'])


class GloVe(object):
    """
    Embeds the article with Glove
    """

    def __init__(self, scribAPI=None, dim=100):
        self.glove = vocab.GloVe(name='6B', dim=dim)
        self.scribAPI = scribAPI
        if scribAPI is not None and scribAPI.config['debug']:
            print('Loaded {} words'.format(len(self.glove.itos)))

    def get_vect_from_word(self, word):
        """
        Give the vector from a word
        """
        return self.glove.vectors[self.glove.stoi[word]]

    def get_word_from_vect(self, vect):
        """
        Give the word from a vector
        """
        for w in self.glove.itos:
            if self.get_vect_from_word(w).numpy().all() == vect.numpy().all():
                return w
        return None

    def tokenize_article_in_words(self, text_article):
        sentences = [word_tokenize(t) for t in sent_tokenize(text_article)]
        words = []
        for sentence in sentences:
            words.extend(sentence)
        return words

    def words_into_vect(self, words):
        vector = None
        for i, word in enumerate(words):
            if i == 0:
                try:
                    vector = self.get_vect_from_word(word.lower())
                except Exception:
                    vector = self.get_vect_from_word('unk')
                    if self.scribAPI is not None and self.scribAPI.config['debug']:
                        print(word.lower())
            elif i == 1:
                try:
                    vector = torch.stack((vector, self.get_vect_from_word(word.lower())), 1)
                except Exception:
                    vector = torch.stack((vector, self.get_vect_from_word('unk')), 1)
                    if self.scribAPI is not None and self.scribAPI.config['debug']:
                        print(words.lower())
            else:
                try:
                    vector = torch.cat((vector, self.get_vect_from_word(word.lower())), 1)
                except Exception:
                    vector = torch.cat((vector, self.get_vect_from_word('unk')), 1)
                    if self.scribAPI is not None and self.scribAPI.config['debug']:
                        print(word.lower())
        return vector

    def __call__(self, sample):
        return torch.transpose(self.words_into_vect(self.tokenize_article_in_words(sample)), 0, 1)

    def reverse(self, vector):
        word = self.get_word_from_vect(vector)
        return word if word is not None else '<unk>'


class ToTensor(object):
    """
    Use Torch tensors
    """
    def __init__(self, word_to_index, start='<s>', end='<e>', unknown='<unk>'):
        self.end = end
        self.start = start
        self.word_to_index = word_to_index
        self.unknown = unknown

    def __call__(self, sample):
        return get_vector_article(sample, self.word_to_index, self.start, self.end, self.unknown)
