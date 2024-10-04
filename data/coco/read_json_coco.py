import json

file = 'dataset_coco.json'
final = []
narrow = 0
intersection = 0
urban = 0

with open(file, 'r') as file_1:
    file_ = json.load(file_1)['images']
    for j in file_:
        if int(j['cocoid']) == 103498:
            list_ = j['sentences']
            for k in list_:
                print(k['raw'])
        














#         if j['Scene Classifier'] == 'Narrow street':
#             narrow += 1
#         elif j['Scene Classifier'] == 'Intersection':
#             intersection += 1
#         elif j['Scene Classifier'] == 'Urban road':
#             urban += 1
# print(narrow / len)
# print(intersection / len)
# print(urban / len)










