import pandas as pd
import sys
import re
import random
import codecs
import six
import unicodedata

ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))


#here we use the pretrain vocab
def count_up(file_name):
    data_temp = pd.read_csv(file_name)

    ttype = ["source","target","language","year","acoustic","genre","context","o"]
    count = dict()
    for _ in ttype:
        count[_] = dict()
        arr = data_temp[_].tolist()
        for i in range(len(arr)):
            
            if(len(data_temp['target'][i])<5):
                continue

            try:
                arr[i] = arr[i].replace(',',' ')

                for word in arr[i].split()[:50]:
                    try:
                        count[_][word] += 1
                    except KeyError:
                        count[_][word] = 1
            except AttributeError:
                continue

    ttype = ["source","target","language","year","acoustic","genre","context","o"]
    for t in ttype:
        arr = [ (n,count[t][n]) for n in count[t]]
        arr = sorted(arr,key= lambda x:x[1],reverse=True)
        count[t] = arr

    ttype = ["source","target","language","year","acoustic","genre","context","o"]
    token = dict()
    for t in ttype:
        token[t] = dict()
        with codecs.open('subword.{0}'.format(t),'w',"utf-8") as f_out:
            #for word token file add up some additional token
            #<PAD> <UNK> <SOS> <EOS>
            if(t == 'source'):
                f_out.write("'<pad>_'\n'<EOS>_'\n") 
            elif(t == 'target'):
                f_out.write("<pad>\n<EOS>\n<SOS>\n") 
            for i in range(len(count[t])):
                f_out.write(u"{0}\n".format(count[t][i][0]))

def tokenlize(file_name):
    ttype = ["source","target","language","year","acoustic","genre","context","o"]
    token = dict()

    random.seed(5)

    for t in ttype:
        token[t] = dict()
        with codecs.open('subword.{0}'.format(t),'r','utf8') as f_in:
            for i,word in enumerate(f_in):

                word = word.strip()
                token[t][word] = str(i)
    
    raw = pd.read_csv(file_name)
    def print_token(f_out,arr,ttype):
        if(isinstance(arr,float)):
            f_out.write(',"')
            f_out.write('"')
            return 
            
        arr = [ token[ttype][str(_)] for _ in arr.replace(',',' ').split()[:50]]
        f_out.write(',"')
        f_out.write(','.join(arr))
        f_out.write('"')

    ttype = ["source","target","language","year","acoustic","genre","context","o"]
    with open("playlist_20181023_train.csv",'w') as f_train:
        with open("playlist_20181023_valid.csv",'w') as f_valid:
            f_train.write('"id","source","target","language","year","acoustic","genre","context","o"\n')
            f_valid.write('"id","source","target","language","year","acoustic","genre","context","o"\n')
            for i in range(raw.shape[0]):
                if(random.random() < 0.90):
                    f_out = f_train
                else:
                    f_out = f_valid
                data = raw.iloc[i]
                
                f_out.write('"{0}"'.format(data["id"]))
                for tt in ttype:
                    print_token(f_out,data[tt],tt)
                f_out.write('\n')

if(__name__ == '__main__'):
    if(sys.argv[1] == 'count'):
        count_up(sys.argv[2])
    elif(sys.argv[1] == 'token'):
        tokenlize(sys.argv[2])

