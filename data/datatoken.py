import pandas as pd
import sys
import re
import random
import codecs

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def count_up():
    max_len = 50

    data_temp = pd.read_csv("./playlist_20180826_parse.csv")

    ttype = ["title_en","title_zh","song","language","year","acoustic","genre","context","o"]
    count = dict()
    for _ in ttype:
        count[_] = dict()
        arr = data_temp[_].tolist()
        for i in range(len(arr)):
            
            if(len(data_temp['song'][i])<5):
                continue

            try:
                arr[i] = arr[i].replace(',',' ')
                if(_ == 'song'):
                    max_len = 48
                else:
                    max_len = 49

                for word in arr[i].split()[:max_len]:
                    try:
                        count[_][word] += 1
                    except KeyError:
                        count[_][word] = 1
            except AttributeError:
                continue

    count['title'] = count['title_en'] 
    
    for name in count['title_zh']:
        try:
            count['title'][name] += count['title_zh'][name]
        except KeyError:
            count['title'][name] = count['title_zh'][name]

    ttype = ["title","song","language","year","acoustic","genre","context","o"]
    for t in ttype:
        arr = [ (n,count[t][n]) for n in count[t]]
        arr = sorted(arr,key= lambda x:x[1],reverse=True)
        count[t] = arr

    count['source'] = count["title"]
    count['target'] = count["song"]

    ttype = ["source","target","language","year","acoustic","genre","context","o"]
    token = dict()
    for t in ttype:
        token[t] = dict()
        with codecs.open('subword.{0}'.format(t),'w',"utf-8") as f_out:
            #for word token file add up some additional token
            #<PAD> <UNK> <SOS> <EOS>
            if(t == 'source' or t == 'target'):
                f_out.write("'<pad>_'\n'<EOS>_'\n'<SOS>_'\n") 
            for i in range(len(count[t])):
                f_out.write(u"'{0}_'\n".format(count[t][i][0]))

def tokenlize():
    ttype = ["source","target","language","year","acoustic","genre","context","o"]
    token = dict()
    max_len = 50

    random.seed(5)

    for t in ttype:
        token[t] = dict()
        with codecs.open('subword.{0}'.format(t),'r','utf8') as f_in:
            for i,word in enumerate(f_in):
                word = word.strip()[1:-2]
                token[t][word] = str(i)
                
    raw = pd.read_csv('playlist_20180826_parse.csv')
    def print_token(f_out,arr,token,ttype,max_len=10000):
        if(isinstance(arr,float)):
            f_out.write(',"')
            f_out.write('"')
            return 
        
        arr = [ token[_] for _ in arr.replace(',',' ').split()[:max_len]]
        f_out.write(',"')
        if(ttype=='target'):
            f_out.write('2,')
        f_out.write(','.join(arr))
        if(ttype=='source' or ttype=='target'):
            f_out.write(',1')
        f_out.write('"')

    ttype = ["title","song","language","year","acoustic","genre","context","o"]
    t_type = ["title_en","title_zh"]
    with open("playlist_20180826_train.csv",'w') as f_train:
        with open("playlist_20180826_valid.csv",'w') as f_valid:
            f_train.write('"id","source","target","language","year","acoustic","genre","context","o"\n')
            f_valid.write('"id","source","target","language","year","acoustic","genre","context","o"\n')
            

            for i in range(raw.shape[0]):
                if(random.random() < 0.90):
                    f_out = f_train
                else:
                    f_out = f_valid
                data = raw.iloc[i]
                
                for t in t_type:
                    f_out.write('"{0}"'.format(data["id"]))
                    print_token(f_out,data[t],token['source'],'source',max_len-1)
                    for tt in ttype[1:]:
                        if(tt=='song'):
                            print_token(f_out,data[tt],token["target"],"target",max_len-2)
                        else:
                            print_token(f_out,data[tt],token[tt],tt)
                    f_out.write('\n')

def test(file_name):
    """
    prepare the data for testing
    """
    ttype = ["title","target","language","year","acoustic","genre","context","o"]
    token = dict()
    max_len = 50

    random.seed(5)


    for t in ttype:
        token[t] = dict()
        with open('token_{0}.label'.format(t)) as f_in:
            for i,word in enumerate(f_in):
                token[t][word.strip()] = str(i)
    

if(__name__ == '__main__'):
    if(sys.argv[1] == 'count'):
        count_up()
    elif(sys.argv[1] == 'token'):
        tokenlize()
    elif(sys.argv[1] == 'test'):
        test(sys.argv[2])

