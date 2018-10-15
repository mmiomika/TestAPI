#from django.shortcuts import render
# Create your views here.
from check.models import Data
from check.serializers import DataSerializer
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt

class DataList(APIView):

    #@api_view(['POST'])
    def post(self,request,format=None):
        data = json.loads(request.body)
        serializer = DataSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    #@api_view(['GET'])
    def get(self,request,format=None):
        check = Data.objects.all()
        serializer = DataSerializer(check, many=True)
        #serializer = DataSerializer()
        return Response(serializer.data)
