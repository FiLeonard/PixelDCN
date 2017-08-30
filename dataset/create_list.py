import glob
import numpy as np

search_path = './train/*'
mask_suffix = '_mask.gif'
data_suffix = '.jpg'

testf = 0.01
validn = 30

names = ['val.txt', 'test.txt', 'train.txt']

all_files = glob.glob(search_path)
all_files = [name for name in all_files if not mask_suffix in name]

l = len(all_files)
testn = np.uint32(testf * l)
np.random.shuffle(all_files)

nlist = [[],[],[]]
nlist[0] = []
nlist[1] = []
nlist[2] = []

for i in range(len(all_files)):
    all_files[i] = all_files[i][1:].replace('\\','/')
    s = ' '.join([all_files[i], all_files[i].replace(data_suffix, mask_suffix)])
    if i < validn:
        idx = 0
    elif i < (testn + validn):
        idx = 1
    else:
        idx = 2
    nlist[idx].append(s)

for j in range(3):
    f = open(names[j], 'w')
    for idx in range(len(nlist[j])):
        f.write(''.join([nlist[j][idx],'\n']))
    f.close
        






