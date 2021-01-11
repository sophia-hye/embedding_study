import pandas as pd
import os

fileName = ['english_23870_first', 'english_23870_all','english_23870_all_merged',
            'english_23870_glove', 'english_23870_w2v']
# english_23870_first: max= 193
# english_23870_all: max=193
# english_23870_all_merged: max= 2161
# english_23870_glove: max=171
# english_23870_w2v: max=144

ROOT_PATH = os.path.join('./dataset/', fileName[3], 'words')

total_list = list(sorted(os.listdir(ROOT_PATH)))
total_list = [os.path.join(ROOT_PATH, l) for l in total_list]

print('start to find max token size')
max = 0
for i, l_name in enumerate(total_list):
    df = pd.read_csv(l_name)
    if max < len(df):
        max = len(df)
    if i%5000 == 0:
        print('finish upto {:5d}/{:5d}'.format(i, len(total_list)))

print('max token size: {}'.format(max)) 
