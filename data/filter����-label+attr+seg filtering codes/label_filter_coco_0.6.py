import json
import os
import time
import gensim

import numpy as np
npy_path = "coco/cocobu_object_name/"
json_file = "../../OpenAI-CLIP-Feature/dataset_coco_retrieval_res101_0830.json"
npy_path_new = "/data2/fjq/1.next_paper/clip_retrieval_filter_new/coco_label_filter_results/new_json_0830/cocobu_label_filter_0.6_res101/"
model_file = 'GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

threshold = 0.6

def compute_similarity(detect_word, label_list):
    result = []
    if detect_word in model:
        for each_word in label_list:
            if each_word in model:
                simi_value = model.similarity(detect_word, each_word)
                if simi_value >= threshold:
                    result.append(detect_word)
            else:
                if detect_word == each_word:
                    result.append(detect_word)
    else:
        if detect_word in label_list:
            result.append(detect_word)

    return result


time1 = time.time()
lst = os.listdir(npy_path)
final = []
with open(json_file, "r", encoding="utf-8") as f:
    file = json.load(f)
    for i in lst:
        if i.endswith(".npy"):
            npy_ = np.load(os.path.join(npy_path + i))
            result_dic = []
            #print(len(file['images']))
            for j in file:
                if int(j["cocoid"]) == int(i[:-4]):
                    print(True)
                    for item_str in npy_:
                        for k in j['sentences']:
                            lis = compute_similarity(item_str, k['tokens'])
                            if len(lis) > 0:
                                result_dic.extend(lis)
                                break
            np.save(os.path.join(npy_path_new + i), result_dic)

time2 = time.time()
print(time2 - time1)


