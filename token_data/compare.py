import pandas as pd
import sys

arr={}
data = pd.read_csv(sys.argv[1])
print(data.shape[0])
for i in range(data.shape[0]):
	temp = data.iloc[i]
	song = [int(_) for _ in temp[2].strip().split(',')]
	arr[temp[1]] = set(song)


beam_size = int(sys.argv[2])

for name in ["pred_eos_output_with_pos","pred_eos_seq2set_no","pred_eos_seq2seq"]:
	stat = [{} for _ in range(beam_size)]
	data = [[] for _ in range(beam_size)]

	for ttype in ['c','g']:
		for lan in ['Chinese','Korean','Western','Japanese','None']:
			
			in_data = [[] for _ in range(beam_size)]
			in_title = []

			f_name = 'to_seasa_20181022/test_kw/{0}_{1}/'.format(ttype,lan)
			with open(f_name+'/beam/'+name) as f:
				for i,line in enumerate(f):
					qq = []
					for _ in line.replace('_',' ').strip().split():
						try:
							qq.append(int(_))
						except:
							pass
					in_data[i%beam_size].append(set(qq))
	
			with open(f_name+'/in.csv') as f:
				for i,line in enumerate(f):
					in_title.append(line.strip())

			print('title',len(in_title))
			print(len(in_data[0]))

			with open(f_name+'/compare/'+name,'w') as f:
				for i in range(len(in_data[0])):
					for j in range(beam_size):
						ans = []
						for title in arr:
							q = len(in_data[j][i] & arr[title])
							if(q > 0):
								ans.append([title,q])
						ans = sorted(ans,key=lambda x:x[1],reverse=True)

						f.write("{0},{1}\n".format(len(in_data[j][i]),in_data[j][i]))
						count = ans[0][1]/len(in_data[j][i])
						
						if(count>0.95):
							count = 0.95
							data[j].append("{0}_{1} {2} ; {3}\n".format(ttype,lan,in_title[i],ans[0][0]))
						elif(count>0.6):
							count = 0.6
						elif(count > 0.3):
							count = 0.3
						else:
							count = 0.1

						try:
							stat[j][count] += 1
						except KeyError:
							stat[j][count] = 1
	
						for t in ans[:3]:
							f.write(str(t[1]/len(in_data[j][i]))[:4]+'|'+str(t[0])+'\n')
						f.write('*'*10)
						f.write('\n')
					f.write('\n')

	with open('./result/'+name,'w') as f:
		for i in range(beam_size):
			for par in [0.95,0.6,0.3,0.1]:
				f.write("{0} {1}\n".format(par,stat[i][par]))
			f.write('\n')
		for i in range(beam_size):
			for j in range(len(data[i])):
				f.write(data[i][j])
				f.write('\n')
			f.write('\n')
