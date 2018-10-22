from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os
import export
from bson.son import SON
from pymongo import DeleteOne

load_dotenv()

CLIENT = MongoClient(os.getenv('DB_ENDPOINT'), 27017)
# CLIENT = MongoClient("mongodb://localhost:27017/")
DB = CLIENT['crypto_hist']

def getDataDf(symbol, tf):
  collection_str = '{0}_{1}'.format(symbol,tf)
  return pd.DataFrame(list(DB[collection_str].find({})))


''' Only for removing duplicate dont try it. it takes very long'''
def remove_duplicate():
  pipeline = [
    {"$group": {"_id": '$t',
                "count": {'$sum': 1},
                "t" :{'$push': '$t'}}},
    {"$match" : {"count": {"$gt": 1}}}
 ]
  for name in DB.collection_names():
    requests = []
    for doc in DB[name].aggregate(pipeline, allowDiskUse= True ):
      print(doc['t'])
      for r in range(len(doc['t']) - 1):
        requests.append(DeleteOne({'t': doc['t'][0]}))
    if(len(requests) > 0):
      DB[name].bulk_write(requests)


def download_all_collections():
  path = 'datasets/crypto_hist/'
  for name in DB.collection_names():
    df = pd.DataFrame(list(DB[name].find({})))
    print('sorting ' + name)
    df = df.sort_values(by='t', ascending=False)
    df.to_csv('{0}{1}.csv'.format(path,name))

def remove_e():
  for name in DB.collection_names():
    DB[name].remove({'t': 'e'})

download_all_collections()