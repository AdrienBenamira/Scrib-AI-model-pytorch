from os import path
import json
import requests


class ScribAPI:
    def __init__(self, file):
        self.config = ScribAPI.import_config(file)
        self.api_endpoint = self.config['url'] + '/api/'

    def get_vocab(self, word=None, limit=None, page=None, number=None, offset=None):
        """
        Get vocab
        :param word: word to get
        :param limit: number of words in total
        :param page: number of the page (0 min)
        :param number: number of words per page
        :param offset:
        :rtype: dict
        """
        if word is not None:
            url = self.api_endpoint + 'vocabulary/one?word=' + word
            params = {}
        else:
            url = self.api_endpoint + 'vocabulary'
            params = {}
            if limit is not None:
                params['limit'] = limit
            if page is not None:
                params['page'] = page
            if number is not None:
                params['number'] = number
            if offset is not None:
                params['offset'] = offset
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def add_word(self, word):
        url = self.api_endpoint + 'vocabulary?word=' + word
        response = requests.post(url)
        response.raise_for_status()
        return response.text

    def get_dataset(self, name, limit=None, page=None, number=None, offset=None):
        """
        Get dataset
        :param name: name of the dataset
        :param limit: number of articles in total
        :param page: number of the page
        :param number: number of articles per page
        :param offset:
        :rtype: dict
        """
        url = self.api_endpoint + 'dataset'
        params = {'name': name}
        if limit is not None:
            params['limit'] = limit
        if page is not None:
            params['page'] = page
        if number is not None:
            params['number'] = number
        if offset is not None:
            params['offset'] = offset
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_dataset_length(self, name):
        """
        Get the size of the given dataset
        :param name:
        :return:
        """
        url = self.api_endpoint + 'dataset/count?dataset=' + name
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def add_article(self, dataset, article, summaries=None):
        """
        Add an article
        :param dataset: dataset linked
        :param article:
        :param summary:
        :return:
        """
        url = self.api_endpoint + "dataset/article"
        params = {
            "dataset": dataset,
            "article": {
                "fullText": article
            }
        }
        if summaries is not None:
            params['summaries'] = [{
                "content": summary
            } for summary in summaries]
        response = requests.post(url, json=params)
        response.raise_for_status()
        return response.text

    def add_dataset(self, dataset):
        url = self.api_endpoint + "dataset?name=" + dataset
        response = requests.post(url)
        response.raise_for_status()
        return response.text

    def add_action(self, article_id, model, content):
        """
            Add a propositon of article by the model
            :param article_id: id of the article 
        """
        url = self.api_endpoint + "model/action"
        params = {
            "model": model,
            "article_id": article_id,
            "content": content
        }
        response = requests.post(url, json=params)
        response.raise_for_status()
        return response.text

    def get_preferences(self, model):
        """
        Get all preferences for training
        :param model: model name
        :return:
        """
        url = self.api_endpoint + 'preference?model=' + model
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def add_preference_number_of_view(self, preference_id):
        """
        Set a preference as treated
        :param preference_id:
        :return:
        """
        url = self.api_endpoint + "preference/treated?id=" + str(preference_id)
        response = requests.post(url)
        response.raise_for_status()
        return response.text

    @staticmethod
    def import_config(file):
        """
        Import config from a json file
        :param file: location relative to the current file
        :return: config
        """
        config_path = path.abspath(path.join(path.dirname(__file__), '..', file))
        with open(config_path, 'r') as config_file:
            config = json.loads(config_file.read())
        config['url'] = 'http://' + config['host'] + ':' + str(config['port'])
        return config
