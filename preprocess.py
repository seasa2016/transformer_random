import codecs

if(__name__ == "__main__"):
    max_len = 50

    input_file = codecs.open('./data/raw_data.csv','r')

    for i,line in enumerate(input_file):
        print(line)
        if(i==5):
            break
