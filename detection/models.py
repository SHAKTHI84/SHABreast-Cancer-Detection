from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class Report(models.Model):
    image = models.ImageField(upload_to='reports/')
    probability = models.FloatField(null=True, blank=True)
    findings = models.TextField(null=True, blank=True)
    guidance = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Report {self.id} - {self.created_at.strftime('%Y-%m-%d')}"

class RiskAssessment(models.Model):
    """Model for breast cancer risk assessment"""
    
    RISK_LEVELS = [
        ('LOW', 'Low Risk'),
        ('MODERATE', 'Moderate Risk'),
        ('HIGH', 'High Risk')
    ]
    
    name = models.CharField(max_length=100, default='Anonymous')
    conversation_history = models.JSONField(default=dict)  # Add default empty dict
    risk_level = models.CharField(
        max_length=10, 
        choices=RISK_LEVELS,
        default='LOW'
    )
    risk_score = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        default=0.0
    )
    key_factors = models.TextField(default='')
    recommendations = models.TextField(default='')
    summary = models.TextField(default='')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Risk Assessment for {self.name} - {self.risk_level}"

    class Meta:
        ordering = ['-created_at']
