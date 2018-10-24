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

def parser(line,name):
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
    return out

final = {'source':{'en':[],'ch':[]},'target':{'en':[],'ch':[]}}

l = {}
with open('casia2015_source.txt') as f_ch:
    with open('casia2015_target.txt') as f_en:
        for i,(l_ch,l_en) in enumerate(zip(f_ch,f_en)):
            l['ch'] = l_ch.replace('_','\\u').strip()
            l['en'] = l_en.replace('_','\\u').strip()
  
            for lan in ['ch','en']:
                for name in ['source','target']:
                    out = parser(l[lan],name)
                    final[name][lan].append(out)


            if(i%100000==0):
                print(i)

#here we finish the chinese to english
#next we add up the language token and the consider four kind
#1.en->ch
#2.ch->en
#3.en->en
#4.ch->ch

#2 '<Chinese\us>_'
#3 '<Taiwanese\us>_'
#4 '<Japanese\us>_'
#5 '<Cantonese\us>_'
#6 '<Korean\us>_'
#7 '<Western\us>_'
#8 '<None\us>_'
#9 '<Mix\us>_'
#10 '<english>_'
#11 '<chinese>_'

data = {"source":[],"target":[]}

for i in range( len( final['source']['ch'] ) ):

    data['source'].append([11] + final['source']['en'][i]) #en diff
    data['source'].append([10] + final['source']['ch'][i]) #ch diff
    data['source'].append([10] + final['source']['en'][i]) #en same
    data['source'].append([11] + final['source']['ch'][i]) #ch same

    data['target'].append(final['target']['ch'][i]) #ch diff
    data['target'].append(final['target']['en'][i]) #en diff
    data['target'].append(final['target']['en'][i]) #en same
    data['target'].append(final['target']['ch'][i]) #ch same


arr = np.random.rand(len(data['source']))
temp = np.array(range(len(arr)))

idx = []
idx.append(temp[arr <= 1.1])
idx.append(temp[arr > 0.98])

print('*'*15)

for i,name in enumerate(['train','test']):
    for lang in ['source','target']:
        with open('pre.{0}.{1}'.format(name,lang),'w') as f_out:
            for id in idx[i]:
                for d in data[lang][id]:
                    f_out.write("{0} ".format(d))
                f_out.write('\n')
        print(i,lang)    




