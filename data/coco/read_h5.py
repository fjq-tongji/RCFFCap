import deepdish as dp

file = r'cocotalk_label.h5'
ff = dp.io.load(file)
h5_total = ff['labels']   ### numpy type
h5_shape = h5_total.shape[0]

min_all = 0
max_all = 100
for i in range(h5_shape):   #### list 16
    min_ = min(h5_total[i])
    max_ = max(h5_total[i])
    if min_ < min_all:
        min_all = min_
    if max_ > max_all:
        max_all = max_

print(min_all)
print(max_all)




label_start_ix = ff['label_start_ix'][:]
label_end_ix = ff['label_end_ix'][:]
print(label_start_ix)
print(label_end_ix)




