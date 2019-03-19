# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:30:50 2018

@author: b99505003
"""

import pandas as pd
import math
import numpy as np
import json
import collections
import itertools

def read_song_meta(fullpath='for_mike_song_meta.txt'):
  data = open(fullpath, 'r').readlines()
  data = [json.loads(d.strip()) for d in data]
  song_meta = {}
  for d in data:
    song_meta[str(d['song_id'])] = d
  return song_meta

def process_tags(tags):
  if not (len(set(tags)) >= 4 or entropy(tags) >= 1.0):
    tag_hist = collections.Counter()
    sum_count = 0
    for tag in tags:
      tag_hist[tag] += 1
      sum_count += 1
    tags = [tag for tag, count in tag_hist.most_common() if float(count)/float(sum_count) >= 0.5]
  return set(tags)

def entropy(labels, base=None):
  """ Computes entropy of label distribution. """
  n_labels = len(labels)
  if n_labels <= 1: return 0
  value, counts = np.unique(labels, return_counts=True)
  probs = counts.astype('float') / n_labels
  n_classes = np.count_nonzero(probs)
  if n_classes <= 1: return 0
  # Compute entropy
  ent = 0.
  base = math.e if base is None else base
  for i in probs:
    ent -= i * math.log(i, base)
  return ent

def intersect(a, b):
  return list(set(a) & set(b))

def load_language_error():
  le = pd.read_csv('editor_playlist_20180419_language_error.csv', header=None)
  le_ids = le.iloc[:,0].tolist()
  le_las = le.iloc[:,1].tolist()
  le = {}
  for le_id, le_la in zip(le_ids, le_las):
    le[str(le_id)] = le_la
  return le

def merge_tags(ptags, mains, subs):
  for sub in subs:
    if sub in ptags:
      ptags.remove(sub)
      for main in mains:
        ptags.append(main)
  return list(set(ptags))
  
song_meta = read_song_meta('20180419_mike_song_meta.csv')

data = pd.read_csv('editor_playlist_20180419_mike.csv', header=None)
pid = data.iloc[:,0].tolist()
# ptitle = data.iloc[:,1].tolist()
ptitle = data.iloc[:,1].tolist()
psongs = data.iloc[:,2].tolist()
ptags = data.iloc[:,3].tolist()
ptags = [pts.split(',') for pts in ptags]

_Chinese = [] #"3": "國語歌曲", 3519, 3116 # 1
_Taiwanese = [] #"10": "台語歌曲", 208, 200 # 3
_Japanese = [] #"17": "日語歌曲", 1043, 1023 # 4, 28
_Cantonese = [] #"24": "粵語歌曲", 113, 111 # 5
_Korean = [] #"31": "韓語歌曲", 1009, 986 # 23
_Indian = [] #"38": "印度語", 4, 4
_Thai = [] #"45": "泰文", 2, 2
_Western = [] #"52": "西洋歌曲", 4064, 3852 # 2
_None = [] # "-1": "非人聲", 941, 892 # 10
_Mix = [] # 484, 1201

plang = []

for index in range(len(pid)):
    songs = psongs[index].split(',')[0:49]
    tags = process_tags([song_meta[s]['language_id'] for s in songs if s in song_meta])
    if len(tags) == 0 or len(tags) > 1:
      _Mix.append(pid[index])
      plang.append('_Mix')
    else:
      tags = list(tags)[0]
      if tags is 3:
        _Chinese.append(pid[index]) #"3": "國語歌曲",
        plang.append('_Chinese')
      elif tags is 10:
        _Taiwanese.append(pid[index]) #"10": "台語歌曲",
        plang.append('_Taiwanese')
      elif tags is 17:
        _Japanese.append(pid[index]) #"17": "日語歌曲",
        plang.append('_Japanese')
      elif tags is 24:
        _Cantonese.append(pid[index]) #"24": "粵語歌曲",
        plang.append('_Cantonese')
      elif tags is 31:
        _Korean.append(pid[index]) #"31": "韓語歌曲",
        plang.append('_Korean')
      elif tags is 38:
        _Indian.append(pid[index]) #"38": "印度語",
        plang.append('_Indian')
      elif tags is 45:
        _Thai.append(pid[index]) #"45": "泰文",
        plang.append('_Thai')
      elif tags is 52:
        _Western.append(pid[index]) #"52": "西洋歌曲",
        plang.append('_Western')
      elif tags is -1:
        _None.append(pid[index]) #-1
        plang.append('_None')
      else:
        print(tags)

if len(pid) == len(plang):
  print('lengths of pid and plang is equal.')

#with open('language_error.csv', 'w') as f:
#  for pi, pn, ps, pt, pl in zip(pid, ptitle, psongs, ptags, plang):
#    # check language
#    if len(intersect(pt, tag_lang)) is not 0:
#      if (pl is '_Chinese' and '1' in pt) or (pl is '_Taiwanese' and '3' in pt) or \
#         (pl is '_Japanese' and ('4' in pt or '28' in pt)) or (pl is '_Cantonese' and '5' in pt) or \
#         (pl is '_Korean' and '23' in pt) or (pl is '_Western' and '2' in pt) or \
#         (pl is '_None') or (pl is '_Indian') or (pl is '_Thai') or (pl is '_Mix'):
#        pass
#      else:
#        f.write('%s,%s,%s,"%s"\nhttps://radio-rdc.kkinternal.com/player/%s\n' % (pi, pl, pn, ','.join(pt), ps))

tag_mapping = pd.read_csv('20180419_mike_tags_mapping.csv', header=None)
tag_id = tag_mapping.iloc[:,0].tolist()
tag_name = tag_mapping.iloc[:,1].tolist()
tag_type = tag_mapping.iloc[:,2].tolist()

tag_lang = []
tag_artist = []
tag_year = []
tag_genre = []
tag_context = []
tag_other = []
tag_name_mapping = {}
tag_type_mapping = {}

for ti, tn, tt in zip(tag_id, tag_name, tag_type):
  tag_name_mapping[str(ti)] = str(tn)
  tag_type_mapping[str(ti)] = str(tt)
  if tt is 'L':
    tag_lang.append(str(ti))
  elif tt is 'A':
    tag_artist.append(str(ti))
  elif tt is 'Y':
    tag_year.append(str(ti))
  elif tt is 'G':
    tag_genre.append(str(ti))
  elif tt is 'C':
    tag_context.append(str(ti))
  elif tt is 'O':
    tag_other.append(str(ti))
  else:
    print(tt)


bad_playlist = ['28545305', '56975543', '33706', '57032179']
le_mapping = load_language_error()
ptag_hist = collections.Counter()

with open('raw_data.csv', 'w') as f:
  with open('no_tags.csv', 'w') as fn:
    f.write('"id","source","song","lang","genre","context"\n')
    for pi, pti, ps, pt, pl in zip(pid, ptitle, psongs, ptags, plang):
      if pi in le_mapping:
          pl = le_mapping[str(pi)]
      if str(pi) not in bad_playlist:
        pt = merge_tags(pt, ['4', '35'], ['28']) 
        # Japanese, Pop # J-POP
        pt = merge_tags(pt, ['12'], ['26', '37', '21022']) 
        # World # Tamil/Hindi, Enka, Latin
        pt = merge_tags(pt, ['19'], ['38']) 
        # Religious # Gospel
        pt = merge_tags(pt, ['21'], ['39', '20123']) 
        # Spoken Word # Audiobook, Storytelling
        pt = merge_tags(pt, ['20408'], ['20198', '20178', '20183', '20188', '20193', '20233', '20344', '20339', '20354']) 
        # Award/Chart # MTV, Oscars, Grammy, BRIT Awards, American Music Awards, Hit Songs, Golden Melody Awards, KKBOX Music Awards, Jade Solid Gold Awards,
        pt = merge_tags(pt, ['6'], ['20208', '20203', '20133', '16', '20213', '20404'])
        # Soundtrack #Movie,TV,Disney,Game,Commercial,Anime
        pt = merge_tags(pt, ['20048'], ['20063', '20058', '18', '20783', '20128', '20118'])
        # Unwinding # Sleep,Café,Spiritual,Music Therapy,Lullaby,Kids Song
        pt = merge_tags(pt, ['21057'], ['20068', '20258', '20763', '20777', '20689'])
        # Romance # Passionate,Valentines Day,Love-Confession,Flirt,Wedding
        pt = merge_tags(pt, ['36'], ['20248', '20735', '20253'])
        # Holiday # New Year, CNY, Christmas
        pt = merge_tags(pt, ['20078'], ['20073', '20083', '20088'])
        # It's Complicated # Heartbroken, Unrequited Love, Long Distance,
        pt = merge_tags(pt, ['20018'], ['20013', '20448', '20023', '21142', '20528'])
        # Workout # Jogging, Cycle, Yoga, BPM, Biking
        pt = merge_tags(pt, ['20028'], ['21131'])
        # Competitions # HBL
        pt = merge_tags(pt, ['20103'], ['20742'])
        # Concentration # Reading
        pt = merge_tags(pt, ['20038'], ['20394'])
        # Happy Hour # Chillout
        pt = merge_tags(pt, ['20770'], ['20756'])
        # Single # Loneliness
        pt = merge_tags(pt, ['22'], ['20917', '20903', '20875', '20882', '20889', '20896'])
        # Rock # Alternative Rock Folk Rock, Pop Rock, Punk Rock, Post Rock, Metal
        pt = merge_tags(pt, ['9'], ['20784', '20854', '20924', '20826', '21128', '20805', '20833', '20819', '20791', '20798', '20812'])
        # Electronic/Dance # EDM, DJ / Breakbeat,Electro/Indie Dance, Drum & Bass, Trap/Twerk, Techno, Trap, Dubstep, Trance, House, Bass
        pt = merge_tags(pt, ['8'], ['20952', '20959', '20966', '20973', '20980', '24'])
        # Jazz # Contemporary Jazz, Vocal Jazz, Latin Jazz / Bossa Nova, Fusion, Swing/ Big Band, Blues
        pt = merge_tags(pt, ['7'], ['20840', '20847'])
        # Hip-Hop/Rap # Conscious Hip-Hop, Alternative Hip-Hop
        pt = merge_tags(pt, ['20703'], ['20931', '20938'])
        # Indie # Indie Pop, Shoegaze/Dream Pop
        pt = merge_tags(pt, ['20'], ['20868'])
        # R&B # Urban R&B

        if '21071' not in pt and '20408' not in pt and '20228' not in pt and '20441' not in pt:
          # ptc = [t for t in pt if (t not in tag_other) and (t not in tag_lang) and (t not in tag_artist) and (t not in tag_year)]
          pt_g = ['g_' + str(g) for g in pt if g in tag_genre]
          pt_c = ['c_' + str(c) for c in pt if c in tag_context]
          for pg in pt_g:
            ptag_hist[pg[2:]] += 1
          for pc in pt_c:
            ptag_hist[pc[2:]] += 1
          if len(pt_g) == 0 and len(pt_c) == 0:
            fn.write('%s,%s,"%s","%s"\n' % (pi, pl, pt, ','.join(pt)))
          if len(pt_g) == 0:
            pt_g.append('_UNK')
          if len(pt_c) == 0:
            pt_c.append('_UNK')
          #for tg, tc in list(itertools.product(pt_g, pt_c)):
          f.write('"%s","%s","%s","%s","%s","%s"\n' % (pi, pti, ps, pl, ','.join(pt_g), ','.join(pt_c)))
        
with open('editor_playlist_20180419_tags_mapping.csv', 'w') as f:
  for tag, count in ptag_hist.most_common():
    f.write('%s,%s,%s,%d\n' % (str(tag), tag_name_mapping[str(tag)], tag_type_mapping[str(tag)], count))
    
