import numpy as np
import sys


#first load the token file
dic= []

with open('subword.{0}'.format(sys.argv[2])) as f:
    for i,line in enumerate(f):
        line = line.strip()
        #remove the ''
        line = line[1:-1]
        
        if(line[-1]=='_'):
            line = line[:-1]

        dic.append(line)

with open(sys.argv[1]) as f:
    for line in f:
        out = [ dic[int(_)] for _ in line.strip().split()]
        print(''.join(out))


