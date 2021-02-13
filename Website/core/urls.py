from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

from . import views


urlpatterns = [
	path('login/', views.loginPage, name="login"),  
	path('logout/', views.logoutUser, name="logout"),
	# path('editprofile/', views.profile, name="profile"),
	# path('profile/<str:name>', views.profilepage, name="profilepage"),
]
