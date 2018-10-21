from django.db import models

# Create your models here.

class Data(models.Model):
    market = models.CharField(max_length=2)
    countryCode = models.CharField(max_length=3)
    userId = models.IntegerField(default=0, null=True)
    clickType = models.CharField(max_length=16)
    clickDate = models.DateTimeField(db_column='clickDate')
    itemId = models.IntegerField()
    page = models.IntegerField(default=1)
    rows = models.IntegerField()

    class Meta:
        db_table = 'clicks_recommend'

class OneItemsCategories(models.Model):
    itemId = models.BigIntegerField(db_column='itemId', primary_key=True)  # Field name made lowercase.
    categoryId = models.BigIntegerField(db_column='categoryId')  # Field name made lowercase.

    class Meta:
        db_table = 'one_items_categories'
        unique_together = (('itemId', 'categoryId'),)


class Items(models.Model):
    itemId = models.IntegerField(db_column='itemId')
    percentItem = models.FloatField(db_column='percentItem')

    class Meta:
        db_table = 'items_recommend'

class Result(models.Model):
    userId = models.IntegerField(default=0, null=True)
    market = models.CharField(max_length=2)
    countryCode = models.CharField(max_length=3)
    categoryId = models.BigIntegerField(db_column='categoryId')
    categoryPercentage = models.FloatField()
    items = models.ManyToManyField(Items)

    class Meta:
        db_table = 'results_recommend'





