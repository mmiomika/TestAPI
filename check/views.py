# Create your views here.
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
from django.db import connection
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from scipy import sparse


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
    '''
    Class for save actual state of recommended items
    '''
    def post(self,request,format=None):
        json_data = request.body.decode('utf-8')
        data1 = json.loads(json_data)
        unix_timestamp = float(data1['clickDate'])/1000
        local_timezone = tzlocal.get_localzone()
        time_format = datetime.fromtimestamp(unix_timestamp, local_timezone)
        data1['clickDate'] = time_format.isoformat()
        data2 = {"itemId": data1['itemId'], "state": data1['state']}
        data1.pop('state')
        serializer = DataSerializer(data=data1)
        serializer1 = ItemsStateSerializer(data=data2)
        if serializer.is_valid():
            serializer.save()
        if serializer1.is_valid():
            serializer1.save()
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
        page = int(test['page'])
        rows = int(test['rows'])
        end = page * rows
        start = end - rows

        # categories

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
        clf = pickle.load(open(os.path.join(BASE_DIR, 'check/model.pkl'), 'rb'))
        categories = clf.predict(fixed_d)
        categories_probability = clf.predict_proba(fixed_d)
        categoryBest = []
        for x in categories_probability:
            categoryBest.append(round(x.max(), 3))
        cats = lb.inverse_transform(categories)

        user_to_item_matrix = pickle.load(open(os.path.join(BASE_DIR, 'check/user_item_matrix.pkl'), 'rb'))
        #sparse.load_npz(os.path.join(BASE_DIR, 'check/user_item_matrix.npz'))
        cosine_similarity_matrix = cosine_similarity(user_to_item_matrix, user_to_item_matrix, dense_output=False)
        cosine_similarity_matrix.setdiag(0)

        # train - table
        cursor = connection.cursor()
        cursor.execute(
            '''
                select distinct *
                   from clicks_recommend
                   where userId is not null and itemId is not null;
            '''
        )
        clicks = cursor.fetchall()
        events = pd.DataFrame(list(clicks), columns=['id', 'market', 'countryCode', 'userId',
                                                     'clickType', 'clickDate', 'itemId', 'page', 'rows'])

        users_list = list(events['userId'])
        current_user = list(test['userId'])[0]

        cursor.execute(
            '''
            select a.itemId, a.state, max(a.updateTime)
            from items_recommend_state as a
            where a.state = '003'
            group by a.itemId, a.state;
            '''
        )
        states = cursor.fetchall()
        state = pd.DataFrame(list(states), columns=['itemId', 'state', 'time'])
        if current_user in users_list:
            user_recommended = cosine_similarity_matrix[current_user].argmax()
            coo_user_matrix = user_to_item_matrix[user_recommended].tocoo()
            sort_df = pd.DataFrame({'itemId': [x for x in coo_user_matrix.col],
                                    'value': [y for y in coo_user_matrix.data]})
            sort_df = sort_df.sort_values(by=['value'], ascending=False).reset_index()
            sort_df = sort_df[['itemId', 'value']]
            sort_df['probability'] = sort_df['value'].apply(lambda x: round(x / sort_df['value'].max(), 3))
            full_df = pd.merge(sort_df, state, how='left', on='itemId')
            full_df = full_df[~pd.isnull(full_df['state'])]
            full_df = full_df.drop_duplicates(['itemId'])
            items = list(full_df['itemId'])
            probab = list(full_df['probability'])
            items_save = items[start:end]
            probab_save = probab[start:end]

            flg = 0
            if len(items) - end >= 0:
                flg = 1

            items_list = []
            for k, v in zip(items_save, probab_save):
                tmp_dict = {"itemId": k, "percentItem": v}
                items_list.append(tmp_dict)

            finalDict = {"market": list(test['market'])[0],
                         "countryCode": list(test['countryCode'])[0],
                         "userId": list(test['userId'])[0],
                         "categoryId": cats[0],
                         "categoryPercentage": categoryBest[0],
                         "isNext": flg,
                         "items": items_list}
            serializer = ResultSerializer(finalDict)
            return JsonResponse(serializer.data)
        else:
            cursor.execute(
                '''
                select b.itemId,
                       b.cnt
                from
                (select a.itemId,
                        a.countryCode,
                        count(a.itemId) as cnt
                 from clicks_recommend as a
                 group by a.itemId, a.countryCode) as b
                where b.countryCode = %s
                order by b.cnt desc;
                ''', [list(test['countryCode'])[0]]
            )
            top_sold = cursor.fetchall()
            top_items = pd.DataFrame(list(top_sold), columns=['itemId', 'cnt'])
            top_items['probability'] = top_items['cnt'].apply(lambda x: round(x / top_items['cnt'].max(), 3))
            full_df = pd.merge(top_items, state, how='left', on='itemId')
            full_df = full_df[~pd.isnull(full_df['state'])]
            full_df = full_df.drop_duplicates(['itemId'])
            items = list(full_df['itemId'])
            probab = list(full_df['probability'])
            items_save = items[start:end]
            probab_save = probab[start:end]

            flg = 0
            if len(items) - end >= 0:
                flg = 1

            items_list = []
            for k, v in zip(items_save, probab_save):
                tmp_dict = {"itemId": k, "percentItem": v}
                items_list.append(tmp_dict)

            finalDict = {"market": list(test['market'])[0],
                         "countryCode": list(test['countryCode'])[0],
                         "userId": list(test['userId'])[0],
                         "categoryId": cats[0],
                         "categoryPercentage": categoryBest[0],
                         "isNext": flg,
                         "items": items_list}
            serializer = ResultSerializer(finalDict)
            return JsonResponse(serializer.data)

        return Response("OK")
    def get(self, request, format=None):
        return Response("WORK WORK WORK", status=status.HTTP_200_OK)


class DataList3(APIView):
    '''
    Class for training of user_item_matrix
    '''

    def get(self, request, format=None):
        return Response("WORK WORK WORK", status=status.HTTP_200_OK)

    def post(self, request, format=None):
        start = datetime.now()
        cursor = connection.cursor()
        cursor.execute(
            '''
                select distinct id as itemId
                from one_items ;
            '''
        )
        items = cursor.fetchall()
        df_items = pd.DataFrame(list(items), columns=['itemId'])
        cursor.execute(
            '''
                select distinct *
                   from clicks_recommend
                   where userId is not null and itemId is not null;
            '''
        )
        clicks = cursor.fetchall()
        events = pd.DataFrame(list(clicks), columns=['id', 'market', 'countryCode', 'userId',
                                                     'clickType', 'clickDate','itemId', 'page', 'rows'])
        n_users = events['userId'].max()
        n_items = df_items['itemId'].max()
        #print(str(n_users) + " " + str(n_items))
        user_to_item_matrix = sparse.dok_matrix((n_users + 1, n_items + 2), dtype=np.int8)

        action_weights = {'CREDIT': 4, 'HOMEPAGE': 3, 'BUYNOW': 5,
                          'PLACEBID': 2, 'SHOWBIDS': 1}

        for row in events.itertuples():
            mapped_user_key = row[4]
            event_type = row.clickType
            if event_type in action_weights.keys():
                user_to_item_matrix[mapped_user_key, row[7]] = action_weights[event_type]
        #sparse.save_npz(os.path.join(BASE_DIR, 'check/user_item_matrix.npz'), user_to_item_matrix)
        pickle.dump(user_to_item_matrix, open(os.path.join(BASE_DIR, 'check/user_item_matrix.pkl'), 'wb'))
        print("Process of training finished. It took {}.".format(datetime.now() - start))
        return Response("TRAIN OK", status=status.HTTP_200_OK)


class DataList4(APIView):
    '''
    Class for training classification
    '''

    def get(self, request, format=None):
        return Response("WORK WORK WORK", status=status.HTTP_200_OK)

    def post(self, request, format=None):
        # start_time = time.time()
        cursor = connection.cursor()
        cursor.execute(
            "(select a.userId, a.market, a.clickType, a.clickDate, a.itemId, a.countryCode, b.categoryId from clicks_recommend a join one_items_categories b on a.itemId = b.itemId order by a.id desc limit 50000) union all (select a.userId, a.market, a.clickType, a.clickDate, a.itemId, a.countryCode, b.categoryId from clicks_recommend a join one_items_categories b on a.itemId = b.itemId order by a.id limit 50000);"
        )
        clicks = cursor.fetchall()
        df = pd.DataFrame(list(clicks),
                          columns=['userId', 'market', 'clickType', 'clickDate', 'itemId', 'countryCode', 'categoryId'])
        print(df.head())
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
        # print("--- %s seconds ---" % (time.time() - start_time))
        return Response("TRAIN OK", status=status.HTTP_200_OK)


class DataList5(APIView):
    '''
    Class for update status
    '''
    def get(self, request, format=None):
        return Response("WORK WORK WORK", status=status.HTTP_200_OK)

    #{"itemId": 697153, "active": true}
    def post(self, request, format=None):
        # start_time = time.time()
        json_data = request.body.decode('utf-8')
        data1 = json.loads(json_data)
        if data1['active'] is True:
            data2 = {"itemId": data1['itemId'], "state": "003"}
        else:
            data2 = {"itemId": data1['itemId'], "state": "005"}
        serializer1 = ItemsStateSerializer(data=data2)
        if serializer1.is_valid():
            serializer1.save()
        return Response("OK", status=status.HTTP_200_OK)



