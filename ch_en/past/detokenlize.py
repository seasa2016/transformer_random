import numpy as np
import sys
import six
import unicodedata

#first load the token file

ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))

dic= []
with open('subword.{0}'.format(sys.argv[2])) as f:
    for i,line in enumerate(f):
        line = line.strip()
        #remove the ''
        line = line[1:-1]
        
        line = line.replace('_',' ').replace('\\u','_')

        dic.append(line)

with open(sys.argv[1]) as f:
    with open(sys.argv[3],'w') as f_out:
        for line in f:
            out = [ dic[int(_)] for _ in line.strip().split()]
            
            ans = '' + out[0]
            for i in range(1,len(out)):
                if((out[i-1][0] in ALPHANUMERIC_CHAR_SET ) != (out[i][0] in ALPHANUMERIC_CHAR_SET )):
                    ans = ans[:-1]    
                ans += out[i]
            
            f_out.write(ans)
            f_out.write('\n')

