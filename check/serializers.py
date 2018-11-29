from rest_framework import serializers
from check.models import Data, OneItemsCategories, Items, Result, ItemsState

class DataSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)
    market = serializers.CharField(max_length=2)
    countryCode = serializers.CharField(max_length=3)
    userId = serializers.IntegerField(default=0, allow_null=True)
    clickType = serializers.CharField(max_length=16)
    clickDate = serializers.DateTimeField()
    itemId = serializers.IntegerField(allow_null=False)
    page = serializers.IntegerField(default=1, allow_null=True)
    rows = serializers.IntegerField(allow_null=True)

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


class OneItemCategoriesSerializer(serializers.Serializer):
    itemId = serializers.IntegerField(read_only=True)
    categoryId = serializers.IntegerField(read_only=True)

    def create(self, validated_data):
        return OneItemsCategories.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.itemId = validated_data.get('itemId', instance.itemId)
        instance.categoryId = validated_data.get('categoryId', instance.categoryId)
        instance.save()
        return instance

class ItemsStateSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)
    itemId = serializers.IntegerField() #PrimaryKeyRelatedField(queryset=ItemsState.objects.all())
    state = serializers.CharField(max_length=3)

    def create(self, validated_data):
        return ItemsState.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.itemId = validated_data.get('itemId', instance.itemId)
        instance.state = validated_data.get('state', instance.state)
        instance.save()
        return instance

class ItemsSerializer(serializers.Serializer):
    itemId = serializers.IntegerField()
    percentItem = serializers.FloatField()

    def create(self, validated_data):
        return Items.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.itemId = validated_data.get('itemId', instance.itemId)
        instance.percentItem = validated_data.get('percentItem', instance.percentItem)
        instance.save()
        return instance

class ResultSerializer(serializers.Serializer):
    userId = serializers.IntegerField(default=0, allow_null=True)
    market = serializers.CharField(max_length=2)
    countryCode = serializers.CharField(max_length=3)
    categoryId = serializers.IntegerField()
    categoryPercentage = serializers.FloatField()
    isNext = serializers.IntegerField(default=0)
    items = ItemsSerializer(many=True)

    class Meta:
        model = Result

    def create(self, validated_data):
        items_data = validated_data.pop('items')
        res = Result.objects.create(**validated_data)

        for item in items_data:
            item, created = Items.objects.get_or_create(itemId=item['itemId'], percentItem=item['percentItem'])
            res.items.add(item)
        return res

    def update(self, instance, validated_data):
        items_data = validated_data.pop('items')

        instance.userId = validated_data.get('userId', instance.userId)
        instance.market = validated_data.get('market', instance.market)
        instance.countryCode = validated_data.get('countryCode', instance.countryCode)
        instance.categoryId = validated_data.get('categoryId', instance.categoryId)
        instance.categoryPercentage = validated_data.get('categoryPercentage', instance.categoryPercentage)
        instance.isNext = validated_data.get('isNext', instance.isNext)

        items_list = []

        for item in items_data:
            item, created = Items.objects.get_or_create(itemId=item['itemId'], percentItem =item['percentItem'])
            items_list.append(item)

        instance.items = items_list
        instance.save()
        return instance










