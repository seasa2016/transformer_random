import sys
with open(sys.argv[1]) as f_in:
	with open(sys.argv[2],'w') as f_out:
		for line in f_in:
			line = line.strip()
			f_out.write("'{0}_'\n".format(line))
