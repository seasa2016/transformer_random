max_len = {}
for ttype in ['source','target']:
    for name in ['train','test']:
        with open('./ch_en.{0}.{1}'.format(name,ttype)) as f:
            
            now = '{0}_{1}'.format(name,ttype)
            
            max_len[now] = 0
            for line in f:
                line = line[:-1].split()
                
                max_len[now] = max(max_len[now],len(line))

for name in max_len:
    print(name,max_len[name])