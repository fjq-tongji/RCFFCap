import json
import os
import time
import gensim

import numpy as np
object_npy_path = "/data1/fjq/1.next_paper/clip_retrieval_filter_new/coco_label_filter_results/cocobu_label_filter_1.0_res101/"
seg_npy_path = "/data1/fjq/bottom-up-attention/coco_segmentation_npy/"
seg_new_npy_path = "/data1/fjq/1.next_paper/clip_retrieval_filter_new/coco_seg_filter_results/cocobu_seg_filter_0.4_res101/"
json_file = "../../OpenAI-CLIP-Feature/dataset_coco_retrieval_res101.json"
model_file = 'GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

threshold = 0.4

def compute_similarity(detect_word, label_list):
    result = []
    if detect_word in model:
        for each_word in label_list:
            if each_word in model:
                simi_value = model.similarity(detect_word, each_word)
                if simi_value >= threshold:
                    result.append(detect_word)
                    #break
            else:
                if detect_word == each_word:
                    result.append(detect_word)
    else:
        if detect_word in label_list:
            result.append(detect_word)

    return result


lst = os.listdir(seg_npy_path)    ### npy list names


with open(json_file, "r", encoding="utf-8") as f:
    file = json.load(f)
    for i in lst:
        time1 = time.time()
        result_dic = []
        object_list = np.load(os.path.join(object_npy_path + i))
        seg_list = np.load(os.path.join(seg_npy_path + i))
        for j in file:
            if int(j["cocoid"]) == int(i[:-4]):
                for item_str in seg_list:
                    for k in j['sentences']:
                        lis1 = compute_similarity(item_str, k['tokens'])
                        if len(lis1) > 0 and (item_str not in object_list):
                            result_dic.extend(lis1)
                            break
                np.save(os.path.join(seg_new_npy_path + i), result_dic)
                break
        time2 = time.time()
        print(time2 - time1)
        print('{} is finished'.format(i))

f.close()




