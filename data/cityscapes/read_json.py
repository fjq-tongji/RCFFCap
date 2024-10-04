import json

file = 'cityscapes_captions4eval.json'
with open(file, 'r') as f:
    k = json.load(f)
    print(k['annotations'])










