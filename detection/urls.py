from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_report, name='upload_report'),
    path('report/<int:pk>/', views.report_detail, name='report_detail'),
    path('chat/', views.chat_view, name='chat'),
    path('risk-assessment/', views.risk_assessment, name='risk_assessment'),
    path('risk-result/<int:pk>/', views.risk_result, name='risk_result'),
    path('analyze-image-chat/', views.analyze_image_chat, name='analyze_image_chat'),
    path('clear-chat/', views.clear_chat, name='clear_chat'),
]