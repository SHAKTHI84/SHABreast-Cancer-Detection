from django.db import models

class Report(models.Model):
    image = models.ImageField(upload_to='reports/')
    probability = models.FloatField(null=True, blank=True)
    findings = models.TextField(null=True, blank=True)
    guidance = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Report {self.id} - {self.created_at.strftime('%Y-%m-%d')}"
