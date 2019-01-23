import numpy as np
import codecs
import sys
import pandas as pd

token = {}
with codecs.open('subword.source','r','utf8') as f:
	for i,word in enumerate(f):
		word = word.strip()[1:-1].replace('_','')
		token[word] = i

for ttype in ['g','c']:
	for i,lan in enumerate(['Western','Chinese','Mix','Korean','Japanese','None']):
		if(lan == 'Mix'):
			continue
		print(ttype,lan)
		f = './to_seasa_20181022/test_kw/{0}_{1}/'.format(ttype,lan)

		ans = pd.read_csv(f+'kw.csv',header=None)
		ans = ans[0].tolist()
		print(len(ans))
		with open(f+'in.csv','w') as f_in:
			with open(f+'data.csv','w') as f_out:
				for line in ans:
					f_in.write(line)
					f_in.write('\n')

					line = line.strip().split()
					f_out.write("{0} ".format(i))
					for word in line:
						f_out.write("{0} ".format(token[word]))
					f_out.write('1\n')


