from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from check import views

urlpatterns = [
    path('check/', views.DataList.as_view()),
]

urlpatterns = format_suffix_patterns(urlpatterns)