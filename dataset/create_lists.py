import glob
import numpy as np

search_path = './prediction/*'
all_files = glob.glob(search_path)

nlist = []

for i in range(len(all_files)):
    all_files[i] = all_files[i][1:].replace('\\','/')
    nlist.append(all_files[i])

f = open("prediction.txt", 'w')
for n in nlist:
    f.write(''.join([n,'\n']))
f.close
print(len(all_files), " filenames written")