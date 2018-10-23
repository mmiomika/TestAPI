# Create your views here.
from check.models import Data, OneItemsCategories
from check.serializers import DataSerializer, ResultSerializer
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
#import time


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
        serializer = DataSerializer(data=data1)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DataList2(APIView):
    def post(self,request,format=None):
        #start_time = time.time()
        json_data = request.body.decode('utf-8')
        data1 = json.loads(json_data)
        unix_timestamp = float(data1['clickDate'])/1000
        local_timezone = tzlocal.get_localzone()
        time_format = datetime.fromtimestamp(unix_timestamp, local_timezone)
        data1['clickDate'] = time_format.isoformat()
        #try:
        #    if data1['']
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
        categoriesDB = OneItemsCategories.objects.values_list()
        list_result = [entry for entry in categoriesDB]
        categoriesDF = pd.DataFrame(list_result, columns=['itemId', 'categoryId'])
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
            result2[k] = sorted_x[start:end]
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
        catList = []
        probab = []
        for x in cats:
            if x in final['categoryId']:
                catList = (list(final[final['categoryId'] == x]['itemsId'])[0])
                probab = (list(final[final['categoryId'] == x]['percentItem'])[0])
        itemsList = []
        for k, v in zip(catList, probab):
            tmpDict = {"itemId": k, "percentItem": v}
            itemsList.append(tmpDict)

        finalDict = {"market": list(test['market'])[0],
                     "countryCode": list(test['countryCode'])[0],
                     "userId": list(test['userId'])[0],
                     "categoryId": cats[0],
                     "categoryPercentage": categoryBest[0],
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
        return Response(serializer.data, status = status.HTTP_200_OK)
