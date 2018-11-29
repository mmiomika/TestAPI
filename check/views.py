# Create your views here.
from check.models import Data, OneItemsCategories, ItemsState
from check.serializers import DataSerializer, ResultSerializer, ItemsStateSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
from datetime import datetime
import tzlocal
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle
from TestAPI.settings import BASE_DIR
from collections import defaultdict
import os
import operator
from django.http import JsonResponse
from django.db import connection, transaction
from scipy import sparse
from matplotlib import dates
from sqlalchemy import create_engine
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def add_missing_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0

def fix_columns( d, columns ):

    add_missing_columns( d, columns )
    assert( set( columns ) - set( d.columns ) == set())
    extra_cols = set( d.columns ) - set( columns )
    if extra_cols:
        print ("extra columns:", extra_cols)

    d = d[ columns ]
    return d

def f(df):
    keys,values=df.sort_values('categoryId').values.T
    ukeys,index=np.unique(keys,True)
    arrays=np.split(values,index[1:])
    df2=pd.DataFrame({'categoryId':ukeys,'itemdId':[list(a) for a in arrays]})
    return df2

class DataList(APIView):

    def post(self,request,format=None):
        json_data = request.body.decode('utf-8')
        data1 = json.loads(json_data)
        print(data1)
        unix_timestamp = float(data1['clickDate'])/1000
        local_timezone = tzlocal.get_localzone()
        time_format = datetime.fromtimestamp(unix_timestamp, local_timezone)
        data1['clickDate'] = time_format.isoformat()
        data2 = {"itemId": data1['itemId'], "state": data1['state']}
        data1.pop('state')
        serializer = DataSerializer(data=data1)
        serializer1 = ItemsStateSerializer(data=data2)
        print(data2)
        if serializer.is_valid():
            serializer.save()
        print(serializer.data)
        if serializer1.is_valid():
            serializer1.save()
        print(serializer1.data)
        #Data.objects.create(data1)
        #ItemsState.objects.create(data2)
        return JsonResponse({"data1": serializer.data, "data2": serializer1.data})

    def get(self, request, format=None):
        return Response("WORK WORK WORK", status=status.HTTP_200_OK)

class DataList2(APIView):
    def post(self,request,format=None):
        #start_time = time.time()
        json_data = request.body.decode('utf-8')
        data1 = json.loads(json_data)
        unix_timestamp = float(data1['clickDate'])/1000
        local_timezone = tzlocal.get_localzone()
        time_format = datetime.fromtimestamp(unix_timestamp, local_timezone)
        data1['clickDate'] = time_format.isoformat()
        test = pd.DataFrame.from_dict(data1, orient='index').T
        lb = LabelEncoder()
        lb.classes_ = np.load(os.path.join(BASE_DIR, 'check/classes.npy'))
        test = test.fillna(0)
        test['weekday'] = time_format.weekday()
        test['month'] = time_format.month
        X_test = test[['itemId', 'countryCode', 'weekday', 'month', 'market']]
        X_test = pd.concat([X_test, pd.get_dummies(X_test['market'],
                                                   prefix="market"),
                            pd.get_dummies(X_test['countryCode'], prefix="countryCode")],
                           axis=1)
        X_test1 = X_test.drop(['countryCode', 'market'], axis=1)
        X = pickle.load(open(os.path.join(BASE_DIR, 'check/trainDataFrame.pkl'), 'rb'))
        fixed_d = fix_columns(X_test1, X.columns)
        clf = pickle.load(open(os.path.join(BASE_DIR,'check/model.pkl'), 'rb'))
        categories = clf.predict(fixed_d)
        categories_probability = clf.predict_proba(fixed_d)
        categoryBest = []
        for x in categories_probability:
            categoryBest.append(round(x.max(),3))
        cats = lb.inverse_transform(categories)
        cursor = connection.cursor()
        cursor.execute(
            "select a.* from qwyIntelect.one_items_categories as a left join qwyIntelect.one_items as b on a.itemId = b.id where b.auctionState <> '015' or b.auctionState <> '008';"
        )
        categoriesDB = cursor.fetchall()
        #categoriesDB = OneItemsCategories.objects.values_list()
        #list_result = [entry for entry in categoriesDB]
        categoriesDF = pd.DataFrame(list(categoriesDB), columns=['itemId', 'categoryId'])
        categoriesDF = categoriesDF[['categoryId', 'itemId']]
        data_category = f(categoriesDF)
        f1 = X['itemId'].value_counts().to_dict()
        result1 = defaultdict(lambda: list(data_category['categoryId']))
        for index, row in data_category.iterrows():
            sender = row['itemdId']
            cat = row['categoryId']
            t = []
            for item in sender:
                t.append(f1.get(item, 0))
            dictionary = {k: v for k, v in zip(sender, t)}
            result1[cat] = dictionary
        page = int(test['page'])
        rows = int(test['rows'])
        end = page * rows
        start = end - rows
        result2 = defaultdict(lambda: list(result1.keys()))
        for k, v in result1.items():
            sorted_x = sorted(v.items(), key=operator.itemgetter(1), reverse=True)
            result2[k] = sorted_x#[start:end]
        final = pd.DataFrame(list(result2.items()), columns=['categoryId', 'items'])
        items_list = [[]]
        prob_list = [[]]
        for v in final['items']:
            tmp = []
            tmp1 = []
            for v1 in v:
                tmp.append(v1[0])
                tmp1.append(v1[1])
            items_list.append(tmp)
            prob_list.append(tmp1)
        items_list.pop(0)
        prob_list.pop(0)
        final['itemsId'] = items_list
        final['probability'] = prob_list
        final = final.drop(['items'], axis=1)
        prob_cnt = [[]]
        for i in final['probability']:
            cnt1 = []
            cnt = 0
            sum1 = sum(i)
            for j in i:
                if sum1 != 0:
                    cnt1.append(round(float(j / sum1), 3))
                else:
                    cnt1.append(round(float(1 / len(i)), 3))
            prob_cnt.append(cnt1)
        prob_cnt.pop(0)
        final['percentItem'] = prob_cnt
        final['percentItem1'] = final['percentItem'].apply(lambda x: x[start:end])
        final['itemsId1'] = final['itemsId'].apply(lambda x: x[start:end])
        final['isNext'] = final['percentItem'].apply(lambda x: 1 if (len(x) - end) >= 0 else 0)
        catList = []
        probab = []
        flg = 0
        for x in cats:
            if x in final['categoryId']:
                catList = (list(final[final['categoryId'] == x]['itemsId1'])[0])
                probab = (list(final[final['categoryId'] == x]['percentItem1'])[0])
                flg = (list(final[final['categoryId'] == x]['isNext'])[0])
        itemsList = []
        for k, v in zip(catList, probab):
            tmpDict = {"itemId": k, "percentItem": v}
            itemsList.append(tmpDict)

        finalDict = {"market": list(test['market'])[0],
                     "countryCode": list(test['countryCode'])[0],
                     "userId": list(test['userId'])[0],
                     "categoryId": cats[0],
                     "categoryPercentage": categoryBest[0],
                     "isNext": flg,
                     "items": itemsList}
        serializer = ResultSerializer(finalDict)
        '''
        SAVE RESULTS OF PREDICT
        if serializer.is_valid():
            serializer.save()
            print("--- %s seconds ---" % (time.time() - start_time))
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        '''
        return JsonResponse(serializer.data)

    def get(self, request, format=None):
        return Response("WORK WORK WORK", status=status.HTTP_200_OK)


class DataList3(APIView):
    def get(self, request, format=None):
        return Response("WORK WORK WORK", status=status.HTTP_200_OK)

    def post(self, request, format=None):
        #start_time = time.time()
        cursor = connection.cursor()
        cursor.execute(
            "(select a.userId, a.market, a.clickType, a.clickDate, a.itemId, a.countryCode, b.categoryId from clicks_recommend a join one_items_categories b on a.itemId = b.itemId order by a.id desc limit 50000) union all (select a.userId, a.market, a.clickType, a.clickDate, a.itemId, a.countryCode, b.categoryId from clicks_recommend a join one_items_categories b on a.itemId = b.itemId order by a.id limit 50000);"
        )
        clicks = cursor.fetchall()
        df = pd.DataFrame(list(clicks), columns=['userId', 'market', 'clickType', 'clickDate', 'itemId', 'countryCode', 'categoryId'])
        print(df.head())
        #df = df.sample(100000)
        df = df.fillna(0)
        df['time'] = pd.to_datetime(df['clickDate'])
        df['weekday'] = df['time'].apply(lambda x: x.weekday())
        df['month'] = df['time'].apply(lambda x: x.month)
        pickle.dump(df, open('trainFull.pkl', 'wb'), protocol=2)
        df_buy = df[df['clickType'] == 'BUYNOW']
        df_hp = df[(df['clickType'] == 'HOMEPAGE')]
        frames = [df_buy, df_hp]
        result = pd.concat(frames)
        X_M = result[['itemId', 'countryCode', 'weekday', 'month', 'market']]
        lb = LabelEncoder()
        lb.fit(result.categoryId)
        y = lb.transform(result.categoryId)
        np.save('classes.npy', lb.classes_)
        X = pd.concat([X_M, pd.get_dummies(X_M['countryCode'], prefix="countryCode"),
                       pd.get_dummies(X_M['market'], prefix="market")], axis=1)
        X = X.drop(['countryCode', 'market'], axis=1)
        pickle.dump(X, open('trainDataFrame.pkl', 'wb'), protocol=2)
        clf = LogisticRegression()
        clf = clf.fit(X, y)
        filename = 'model.pkl'
        pickle.dump(clf, open(filename, 'wb'), protocol=2)
        #print("--- %s seconds ---" % (time.time() - start_time))
        return Response("TRAIN OK", status=status.HTTP_200_OK)


