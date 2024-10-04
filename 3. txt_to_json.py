
list_result = []     ### 一张图像存成一个字典，3张图像-3个字典，放到一个list中
all_img_id = []
all_captions = []


txt_path = r"E:\博一上学期\2. 图像描述论文实战\自己的数据集cityscapes\id+caption.txt"

path_captions  = r'E:\博一上学期\2. 图像描述论文实战\自己的数据集cityscapes\captions.txt'
path_train_img = r'E:\博一上学期\2. 图像描述论文实战\自己的数据集cityscapes\cityscapes_train_list.txt'

captions_files = open(path_captions, encoding='utf-8')
img_id = open(path_train_img, encoding='utf-8')

captions_lines = captions_files.readlines()  # 读取所有内容
captions_img_num = len(captions_lines) // 5          ### 总共标记了多少张图片
print(captions_img_num)

for line in img_id.readlines():    #### 所有的图片名，列表
    line = line.strip("\n")
    all_img_id.append(line)

captions_lines = captions_lines

for line in captions_lines:  #### 所有的标注句子，列表
    line = line.strip("\n")
    all_captions.append(line)
#print(all_captions)

for i in range(captions_img_num):
    dict={}
    dict['file_path'] = all_img_id[i]
    dict['captions'] = []
    for j in [i*5, i*5+1, i*5+2, i*5+3, i*5+4]:
        dict['captions'].append(all_captions[j])

    list1 = all_img_id[i].split('_')
    #print(list1)
    id = list1[0] + '_' + list1[1] + '_' + list1[2]
    dict['id'] = id

    #dict['split'] = 'train'
    list_result.append(dict)

print(list_result)


import json

filename = r'E:\博一上学期\2. 图像描述论文实战\自己的数据集cityscapes\cityscapes.json'

with open(filename, 'w') as file_obj:
    json.dump(list_result, file_obj)