import h5py

file_path = 'cityscapes_paragraph_trainval_talk.h5'
with h5py.File(file_path, 'r') as file:
    # 获取数据集（dataset）
    dataset_name = 'cityscapes_paragraph_trainval_talk'  # 替换为你的数据集名称
    data = file.visit(lambda x: print(x))
    print(data)
    data = file['None'][:]  # 读取整个数据集

    # 进行你的操作，比如打印数据
    print(data)
