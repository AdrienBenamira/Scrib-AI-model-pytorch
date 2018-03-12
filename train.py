from utils import ScribAPI, ScribDataset

scribAPI = ScribAPI('./config/local.json')

preferences = scribAPI.get_preferences('onmt')
# result = scribAPI.add_preference_number_of_view(8)
print(len(preferences))
