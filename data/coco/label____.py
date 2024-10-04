import json
import os
import numpy as np
import time
json_file_3 = "dataset_coco_retrieval_res101_new.json"
json_file_1 = "dataset_coco.json"

with open(json_file_1, "r", encoding="utf-8") as f:
    file = json.load(f)
    #for i in file:
     #   print(file.keys())
    #print(len(file['images']))
    for j in file['images']:
        if int(j['cocoid']) == 6701:
            print(j)


