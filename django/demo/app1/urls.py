from django.urls import path, re_path
from app1 import views

urlpatterns = [
    #path('articles/2018', views.article),
    #path('articles/<int:year>', views.article),
    #re_path(r'articles/(?P<year>[0-9]{4})', views.article),
    re_path(r'^articles/(?P<year>[0-9]{4})/$', views.article),
    path('get_name', views.get_name),
    path('get_name_class', views.PersonFormView.as_view()),
    path('person_detail/<int:pk>', views.person_detail),
]