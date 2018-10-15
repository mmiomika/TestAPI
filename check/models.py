from django.db import models

# Create your models here.

class Data(models.Model):
    market = models.CharField(max_length=2)
    countryCode = models.CharField(max_length=3)
    userId = models.IntegerField(default=0, null=True)
    clickType = models.CharField(max_length=16)
    clickDate = models.DateField('clickDate')
    itemId = models.IntegerField()
    page = models.IntegerField(default=1)
    rows = models.IntegerField()

    class Meta:
        db_table = '"check_data_test"'



