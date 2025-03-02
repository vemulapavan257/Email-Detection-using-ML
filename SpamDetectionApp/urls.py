from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('Login.html', views.Login, name="Login"), 
	       path('UserLogin', views.UserLogin, name="UserLogin"),
	       path('UploadDataset.html', views.UploadDataset, name="UploadDataset"),
	       path('UploadDatasetAction', views.UploadDatasetAction, name="UploadDatasetAction"),
	       path('TrainData.html', views.TrainData, name="TrainData"),
	       path('SpamDetection.html', views.SpamDetection, name="SpamDetection"),
	       path('SpamDetectionAction', views.SpamDetectionAction, name="SpamDetectionAction"),	      
	       path('TrainDataGA', views.TrainDataGA, name="TrainDataGA"),
]