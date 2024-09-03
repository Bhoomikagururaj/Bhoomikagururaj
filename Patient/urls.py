from django.urls import path
from .import views 


urlpatterns=[ 
    path('',views.index,name='home'),
    path("register",views.register,name='register'),
    path("login",views.login,name='login'),
    path("dataentry",views.dataentry,name='dataentry'),
    path("prediction",views.prediction,name='prediction'),
    path('logout',views.logout,name='logout')

]