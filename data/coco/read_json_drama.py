import json

file = 'dataset_coco.json'
final = []
narrow = 0
intersection = 0
urban = 0

with open(file, 'r') as file:
    file_ = json.load(file['images'])
    len = len(file_)
    print(file_)














#         if j['Scene Classifier'] == 'Narrow street':
#             narrow += 1
#         elif j['Scene Classifier'] == 'Intersection':
#             intersection += 1
#         elif j['Scene Classifier'] == 'Urban road':
#             urban += 1
# print(narrow / len)
# print(intersection / len)
# print(urban / len)










