from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from check import views

urlpatterns = [
    path('save/', views.DataList.as_view()),
    path('predict/', views.DataList2.as_view()),
    path('train/', views.DataList3.as_view()),
    path('trainCategories/', views.DataList4.as_view()),
]

urlpatterns = format_suffix_patterns(urlpatterns)