import os
from ScribAPI import ScribAPI
import requests
from utils import tokenize
from datasets.ScribDataset import ScribDataset


def get_article(file):
    article = ""
    line = file.readline()
    while line != '@highlight\n':
        article += line
        line = file.readline()
    return article


def import_articles():
    """
    Add the articles to
    :return:
    """
    datasets = ['cnn', 'daily']
    scribAPI = ScribAPI('./config/local.json')
    for dataset in datasets:
        # relative path
        folder_path = '../../Scrib-AI/AI/data/' + dataset + '/stories/'
        # absolute path
        folder_path = os.path.abspath(os.path.join(__file__, folder_path))
        print("Saving", dataset)
        for _, _, files in os.walk(folder_path):
            # print(files)
            len_files = len(files)
            print(len_files, 'to save...')
            for k, file in enumerate(files):
                print(k / len_files * 100, '%')
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'r') as story:
                    article = get_article(story)
                    try:
                        scribAPI.add_article('cnn-daily', article)
                    except requests.exceptions.HTTPError:

                        print(file_path, 'failed')
                    else:
                        print(file_path, 'saved!')


def import_vocab():
    """
    Import the vocab from cnn-daily model
    :return:
    """
    scribAPI = ScribAPI('./config/local.json')
    cnn_dataset = ScribDataset(scribAPI, 'cnn-daily')
    for i in range(len(cnn_dataset)):
        print(i, 'over', len(cnn_dataset))
        words = tokenize(cnn_dataset[i]['article'][0])
        for word in words:
            scribAPI.add_word(word)


if __name__ == '__main__':
    import_vocab()
