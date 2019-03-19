import numpy as np
import codecs
import sys
import six
import unicodedata


ALPHANUMERIC_CHAR_SET = set(
	six.unichr(i) for i in range(sys.maxunicode)
	if (unicodedata.category(six.unichr(i)).startswith("L") or
	unicodedata.category(six.unichr(i)).startswith("N")))
token = {}
with codecs.open('subword.source','r','utf8') as f:
	for i,word in enumerate(f):
		word = word.strip()
		token[word[1:-1]] = i

def convert(line):
	out = []
	start = 0
	end = len(line)
	while(start < end):
		for mid in range(end,start,-1):
			try:
				out.append(token[line[start:mid]])
				start = mid
				break
			except KeyError:
				pass
	return out


for ttype in ['g','c']:
	for i,lan in enumerate(['Chinese','Taiwanese','Japanese','Cantonese','Korean','Western','None']):
		if(lan == 'Taiwanese' or lan == 'Cantonese' or lan == 'Mix'):
			continue
		print(ttype,lan)
		f = './to_seasa_20181022/test_kw/{0}_{1}/'.format(ttype,lan)
		with open(f+'in.csv') as f_in:
			with open(f+'data.csv','w') as f_out:
				for line in f_in:
					line = line.replace('_','\\u').strip().split()
					out = []            
					for word in line:
						data = [ _ in ALPHANUMERIC_CHAR_SET for _ in word ]
						start = 0
						for j in range(1,len(word)):
							if(data[j] != data[j-1]):
								out.extend(convert("{0}_".format(word[start:j])))
								start = j
						else:
							out.extend(convert("{0}_".format(word[start:])))
					f_out.write("{0} ".format(i))
					for num in out:
						f_out.write("{0} ".format(num))
					f_out.write('\n')


