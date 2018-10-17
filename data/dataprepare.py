import codecs
import pandas as pd
import os
import jieba,re,string
from googletrans import  Translator
translator = Translator()

def jieba_setting():
    jieba.set_dictionary('./jieba_dict/dict.txt.big')
    jieba.load_userdict('./jieba_dict/20180419-artistname_zh_tw.txt')
    jieba.load_userdict('./jieba_dict/20180419-albumname_zh_tw.txt')
    jieba.load_userdict('./jieba_dict/20180419-songname_zh_tw.txt')
    jieba.load_userdict('./jieba_dict/20180425-en-dict.txt')

    with open('./jieba_dict/editor_playlists-20161020-dict.txt', 'r', encoding='utf-8') as f:
        for _ in f:
            jieba.add_word( _.split(',')[0])
            
    stopword_set = set()
    with open('./jieba_dict/stopwords.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
    with open('./jieba_dict/stopwords2.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
    with open('./jieba_dict/stopwords_internet.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
    return stopword_set    

def sen_pre_processing(sen):
    sen = str(sen).lower()
    sen = re.sub(r'[\u3000-\u303F]', ' ', sen)
    sen = re.sub(r'[\uFF00-\uFFEF]', ' ', sen)

    emoji = '💕' + '🤘' + '👯' + '🏋️' + '🚗🚗' + '💪' + '🎄' + '🔁' + '👋' + '💪💪' + '🔥' + \
          '😝👶🏻☀️🍹' + '🔁' + '✨👯' + '🌞' + '💖' + '🎉✨' + '🔁' + '🌸' + '❤️💛💚💙💜' + \
          '🇫🇷' + '🎆' + '💋' + '🔁' + '👂' + '🎸' + '💦💦' + '💦💦' + '❤' + '🍷' + \
          '🎂' + '👫' + '🌞' + '🌊' + '🍁' + '🔁' + '🎶' + '🌹🌹' + '😆' +'🐔' + '🤠🇹🇭' + \
          '👯' + '🌙' + '🎄' + '🎤' + '🔥🔥' + '🔥🔥' + '📚' + '💕' + '🙋' + '😁👾🍸' + '🔥' + \
          '🎓' + '🍷' + '✏️🍍🍎✏️' + '💧' + '🔁' + '🌞' + '🛋' + '🔁' + '💪🏼' + '💔' + \
          '🔁' + '🏆' + '🏆' + '🔁' + '🎉' + '🎉' + '🏃👟' + '🍖🍷🍾' + '❤️' + '🔁' + '😢' + \
          '👋🏼' + '👂' + '🗽' + '🇵🇷' + '🎸' + '🔁' + '🌧' + '🔁' + '🍿🍿' + '❤️' + '😈' + \
          '😈' + '🇭🇰' + '🌞' + '🌞' + '😜' + '' + '' + '' + '' + ''
    for c in string.punctuation + emoji+'”“':
        sen = sen.replace(c, ' ')

    return sen

#stopword_set = jieba_setting()

tag = dict()
with open('./20180419_mike_tags_mapping.csv') as f:
    for line in f:
        line = line.strip().split(',')
        tag[int(line[0])] = line[2]
raw_data = pd.read_csv('./editor_playlist_20180419_mike.csv', header=None)

data = dict()

print('Start...')

for i in range(raw_data.shape[0]):
    data_temp = raw_data.iloc[i]
    p_id = data_temp[0]
    title = bytes(data_temp[1], 'utf-8').decode('utf-8', 'igonre')
    song = [int(_) for _ in data_temp[2].strip().replace(',',' ').split()]
    p_type = [int(_) for _ in data_temp[3].strip().replace(',',' ').split()]

    if len(title) >= 0:
        output = dict()
        print(title)
        title  = sen_pre_processing(title)
        translator = Translator()
        try:
            output['zh-tw'] = translator.translate(title, dest='zh-tw',src='en').text
            output['en'] = translator.translate(title, dest='en',src='zh-tw').text
        except:
            translator = Translator()
            output['zh-tw'] = translator.translate(title, dest='zh-tw',src='en').text
            output['en'] = translator.translate(title, dest='en',src='zh-tw').text
            
        try:
            if(isinstance(data[p_id],dict)):
                print('repeat',p_id)
        except KeyError:
            pass
        
        data[p_id] = dict()
        data[p_id]['title'] = [output['zh-tw'],output['en']]
        data[p_id]['song'] = song
        data[p_id]['language'] = []
        data[p_id]['year'] = []
        data[p_id]['acoustic'] = []
        data[p_id]['genre'] = []
        data[p_id]['context'] = []
        data[p_id]['o'] = []
        
        for t in p_type:
            if(tag[t] == 'L'):
                data[p_id]['language'].append(t)
            elif(tag[t] == 'Y'):
                data[p_id]['year'].append(t)
            elif(tag[t] == 'A'):
                data[p_id]['acoustic'].append(t)
            elif(tag[t] == 'C'):
                data[p_id]['context'].append(t)
            elif(tag[t] == 'G'):
                data[p_id]['genre'].append(t)
            elif(tag[t] == 'O'):
                data[p_id]['o'].append(t)
        if i % 100 == 0:
            print('{0}'.format(i))

with open('playlist_20180826_parse.csv','w') as f:
    f.write('"id","title_en","title_zh","song","language","year","acoustic","genre","context","o"\n')
    for i in data:
        txt = '"{0}","{1}","{2}","{3}","{4}","{5}","{6}","{7}","{8}"\n'.format(
                str(i),
                data[i]['title'][1],
                ','.join([ str(_) for _ in data[i]['song']]),
                ','.join([ str(_) for _ in data[i]['language']]),
                ','.join([ str(_) for _ in data[i]['year']]),
                ','.join([ str(_) for _ in data[i]['acoustic']]),
                ','.join([ str(_) for _ in data[i]['genre']]),
                ','.join([ str(_) for _ in data[i]['context']]),
                ','.join([ str(_) for _ in data[i]['o']])
            )
        f.write(txt)

        txt = '"{0}","{1}","{2}","{3}","{4}","{5}","{6}","{7}","{8}"\n'.format(
                str(i),
                data[i]['title'][0],
                ','.join([ str(_) for _ in data[i]['song']]),
                ','.join([ str(_) for _ in data[i]['language']]),
                ','.join([ str(_) for _ in data[i]['year']]),
                ','.join([ str(_) for _ in data[i]['acoustic']]),
                ','.join([ str(_) for _ in data[i]['genre']]),
                ','.join([ str(_) for _ in data[i]['context']]),
                ','.join([ str(_) for _ in data[i]['o']])
            )
        f.write(txt)
    