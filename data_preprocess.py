import os
import codecs
df_q__ = df_q_[df_q_['flags'].str.contains('SUCCESS')].copy()
path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/finaldata'
newpath = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/finaldata_preprocessed'
for file in os.listdir(path):
    f=open(os.path.join(path, file), 'r+')
    data = f.readlines()
    fnew = open(os.path.join(newpath, file), 'w+')
    if file.split('.')[-1]=='txt':
        fnew.writelines(data)
        f.close()
        fnew.close()
    else:
        newdata = [line for line in data if len(line.strip())!=0]
        newdata_filter = []
        for id, line in enumerate(newdata):
            # if '[' in line and '[' in line:
            #     continue
            # if '?' in line:
            #     continue
            # if '(' and ')' in line:
            #     continue
            if '---' in line:
                print(file)
                print(line)
                temp_idx = int(newdata_filter[id-1].split()[3])-95
                temp = newdata_filter[id-1].split()
                newdata_filter[id - 1] = ' '.join([temp[0],temp[1],temp[2],str(temp_idx),temp[4],'\n'])
                #continue
            newdata_filter.append(line)
        fnew.writelines([item for item in newdata_filter if '---' not in item])
        f.close()
        fnew.close()


