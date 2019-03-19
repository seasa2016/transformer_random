import numpy as np


with open('ch_en.test2.source') as f_1:
    num = len(f_1.readlines())

arr = np.random.rand(num)

with open('ch_en.test2.source') as f_1:
    with open('ch_en.test.source','w') as f_valid:
        with open('ch_en.valid.source','w') as f_test:
            for i,line in enumerate(f_1):
                if(arr[i]<0.99):
                    f_valid.write(line)
                else:
                    f_test.write(line)

with open('ch_en.test2.target') as f_1:
    with open('ch_en.test.target','w') as f_valid:
        with open('ch_en.valid.target','w') as f_test:
            for i,line in enumerate(f_1):
                if(arr[i]<0.99):
                    f_valid.write(line)
                else:
                    f_test.write(line)