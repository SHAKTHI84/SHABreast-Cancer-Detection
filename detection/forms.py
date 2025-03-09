from django import forms
from .models import Report, RiskAssessment

class ReportForm(forms.ModelForm):
    class Meta:
        model = Report
        fields = ['image']

class RiskAssessmentForm(forms.ModelForm):
    class Meta:
        model = RiskAssessment
        exclude = ['risk_score', 'recommendations', 'created_at']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Bootstrap classes
        for field in self.fields:
            self.fields[field].widget.attrs.update({
                'class': 'form-control mb-3'
            })
            if isinstance(self.fields[field], forms.BooleanField):
                self.fields[field].widget.attrs['class'] = 'form-check-input mb-3'