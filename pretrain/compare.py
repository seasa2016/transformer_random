import pandas as pd
import sys

arr={}
data = pd.read_csv(sys.argv[1])
print(data.shape[0])
for i in range(data.shape[0]):
	temp = data.iloc[i]
	song = [int(_) for _ in temp[2].strip().split(',')]
	arr[temp[1]] = set(song)

stat = {}
data = []

for ttype in ['c','g']:
	for lan in ['Chinese','Korean','Western','Japanese','None']:
		in_data = []
		f_name = 'to_seasa_20181022/test_kw/{0}_{1}/'.format(ttype,lan)
		with open(f_name+'/'+sys.argv[2]) as f:
			for line in f:
				in_data.append([set([int(_) for _ in line.replace('_',' ').strip().split()[:-1]]),0])

		with open(f_name+'/in.csv') as f:
			for i,line in enumerate(f):
				in_data[i][1] = line.strip()

		with open(f_name+'/compare_data_'+sys.argv[2],'w') as f:
			for i in range(len(in_data)):
				ans = []
				for name in arr:
					q = len(in_data[i][0] & arr[name])
					if(q > 0):
						ans.append([name,q])
				ans = sorted(ans,key=lambda x:x[1],reverse=True)
				f.write("{0},{1}\n".format(len(in_data[i][0]),in_data[i][1]))
				count = ans[0][1]/len(in_data[i][0])
				
				if(count>0.95):
					count = 0.95
					data.append("{0}_{1} {2} ; {3}\n".format(ttype,lan,in_data[i][1],ans[0][0]))
				elif(count>0.6):
					count = 0.6
				elif(count > 0.3):
					count = 0.3
				else:
					count = 0.1
				try:
					stat[count] += 1
				except KeyError:
					stat[count] = 1



				for t in ans[:3]:
					f.write(str(t[1]/len(in_data[i][0]))[:4]+'|'+str(t[0])+'\n')
				f.write('*'*10)
				f.write('\n')
with open(sys.argv[2],'w') as f:
	for name in stat:
		f.write("{0} {1}\n".format(name,stat[name]))
	for line in data:
		f.write(line)
