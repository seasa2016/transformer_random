dic = dict()

with open('./casia2015_ch.txt') as f:
    for line in f:
        for cha in line:
            try:
                dic[cha] += 1
            except KeyError:
                dic[cha] = 1
arr = [(name,dic[name]) for name in dic]

with open('./casia2015_ch_count.txt','w') as f:
    for i in arr:
        f.write('{0} {1}\n'.format(i[0],i[1]))

