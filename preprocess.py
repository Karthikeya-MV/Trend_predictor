# train_model.py
import pandas as pd
import os

l = os.listdir('Data')
data = {'hashtag':[],'count':[]}
for i in l:
    df = pd.read_csv('Data/'+i)
    for j in range(len(df)):
        data['hashtag'].append(df['trend_name'][j])
        data['count'].append(df['tweet_volume'][j])
data = pd.DataFrame(data).fillna(1)
data.to_csv('Data/twitter_data.csv',index=False)
