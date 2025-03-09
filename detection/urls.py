from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_report, name='upload_report'),
    path('report/<int:pk>/', views.report_detail, name='report_detail'),
    path('chat/', views.chat_view, name='chat'),
]