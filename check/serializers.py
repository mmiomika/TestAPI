from rest_framework import serializers
from check.models import Data

class DataSerializer(serializers.Serializer):
    #class Meta:
    #    model = Data
    #    fields = ('id', 'market', 'countryCode', 'userId', 'clickType', 'clickDate', 'itemId', 'page', 'rows')
    id = serializers.IntegerField(read_only=True)
    market = serializers.CharField(max_length=2)
    countryCode = serializers.CharField(max_length=3)
    userId = serializers.IntegerField(default=0, allow_null=True)
    clickType = serializers.CharField(max_length=16)
    clickDate = serializers.DateField('clickDate')
    itemId = serializers.IntegerField()
    page = serializers.IntegerField(default=1)
    rows = serializers.IntegerField()

    def create(self, validated_data):
        return Data.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.market = validated_data.get('market', instance.market)
        instance.countryCode = validated_data.get('countryCode', instance.countryCode)
        instance.userId = validated_data.get('userId', instance.userId)
        instance.clickType = validated_data.get('clickType', instance.clickType)
        instance.clickDate = validated_data.get('clickDate', instance.clickDate)
        instance.itemId = validated_data.get('itemId', instance.itemId)
        instance.page = validated_data.get('page', instance.page)
        instance.rows = validated_data.get('rows', instance.rows)
        instance.save()
        return instance


