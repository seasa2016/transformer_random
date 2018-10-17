import sys
temp=[]
with open(sys.argv[1]) as f:
    for line in f:
        temp.append(line.lower())

with open(sys.argv[1],'w') as f:
    for line in temp:
        f.write("{0}".format(line))
