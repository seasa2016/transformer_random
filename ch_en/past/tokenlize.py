import numpy as np
import unicodedata
import six
import sys

np.random.seed(36)

ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))


#first load the token file
dic={'source':{},'target':{}}

for name in dic:
    with open('subword.{0}'.format(name)) as f:
        for i,line in enumerate(f):
            line = line.strip()
            #remove the ''
            line = line[1:-1]
            
            dic[name][line] = i

def convert(line,name):
    out = []
    start = 0
    end = len(line)

    while(start < end):
        for mid in range(end,start,-1):
            try:
                out.append(dic[name][line[start:mid]])
                start = mid
                break
            except KeyError:
                pass
    return out

final = {'source':[],'target':[]}
for name in dic:
    with open('casia2015_{0}.txt'.format(name)) as f:
        for i,line in enumerate(f):
            line = line.replace('_','\\u').strip()
            out = []            
            for word in line.split():
                #print("word",word)
                data = [ _ in ALPHANUMERIC_CHAR_SET for _ in word ]
                #print(data)
                start = 0
                for j in range(1,len(word)):
                    if(data[j] != data[j-1]):
                        out.extend(convert("{0}_".format(word[start:j]),name))
                        start = j
                else:
                    out.extend(convert("{0}_".format(word[start:]),name))
            final[name].append(out)
            if(i%100000==0):
                print(i)
            
arr = np.random.rand(len(final['source']))
temp = np.array(range(len(arr)))


idx = []
idx.append(temp[arr <= 0.8])
idx.append(temp[arr > 0.8])

for i,name in enumerate(['train','test']):
    for lang in ['source','target']:
        with open('ch_en.{0}.{1}'.format(name,lang),'w') as f_out:
            for id in idx[i]:
                for d in final[lang][id]:
                    f_out.write("{0} ".format(d))
                f_out.write('\n')




