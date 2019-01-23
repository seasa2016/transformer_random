import numpy as np


with open('test2.source') as f_1:
    num = len(f_1.readlines())

arr = np.random.rand(num)

with open('test2.source') as f_1:
    with open('test.source','w') as f_valid:
        with open('valid.source','w') as f_test:
            for i,line in enumerate(f_1):
                if(arr[i]<0.99):
                    f_valid.write(line)
                else:
                    f_test.write(line)

with open('test2.target') as f_1:
    with open('test.target','w') as f_valid:
        with open('valid.target','w') as f_test:
            for i,line in enumerate(f_1):
                if(arr[i]<0.99):
                    f_valid.write(line)
                else:
                    f_test.write(line)