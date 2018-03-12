from utils import ScribAPI, ScribDataset

config_file = './config/local.json'

scribAPI = ScribAPI(config_file)

"""
Get preferences from the database.

Dict format is a list of preferences dict. There is the format of the dicts:
* id
* score:
  * -1 none of the choices is good
  * 0 both choices are good
  * 1 left action is better
  * 2 right action is better
* treated: number of time you requested this preference
* action_right dict:
  * id
  * content: summary content
  * Article dict
    * fullText: article
* action_left dict:
  * id
  * content: summary content
  * Article dict
    * fullText: article
"""
preferences = scribAPI.get_preferences(scribAPI.config['model']['name'])
print(preferences)
