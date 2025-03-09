from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Report
from .forms import ReportForm
from langchain_community.llms import Ollama
import base64
import json

# Configure Ollama
llm = Ollama(
    model="llama3.2-vision",
    base_url="https://9gwg2slc-11434.inc1.devtunnels.ms/",
    temperature=0.1
)

def upload_report(request):
    if request.method == 'POST':
        form = ReportForm(request.POST, request.FILES)
        if form.is_valid():
            report = form.save()
            # Process with Llama model
            results = analyze_image(report.image.path)
            report.probability = results['probability']
            report.findings = results['findings']
            report.guidance = results['guidance']
            report.save()
            return redirect('report_detail', pk=report.pk)
    else:
        form = ReportForm()
    return render(request, 'detection/upload.html', {'form': form})

def report_detail(request, pk):
    report = Report.objects.get(pk=pk)
    return render(request, 'detection/detail.html', {'report': report})

def chat_view(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        response = get_chatbot_response(question)
        return JsonResponse({'response': response})
    return render(request, 'detection/chat.html')

def analyze_image(image_path):
    try:
        # Read and encode image
        with open(image_path, 'rb') as img:
            image_data = base64.b64encode(img.read()).decode('utf-8')
        
        # Create analysis prompt with proper JSON structure
        system_prompt = """You are a medical AI assistant specialized in analyzing mammograms.
        Analyze the provided image and give a structured response about potential breast cancer indicators."""
        
        user_prompt = """Please analyze this mammogram and provide:
        1. A percentage indicating the probability of cancer (0-100)
        2. Key findings from the image
        3. Medical guidance for the patient
        
        Format the response as a structured analysis."""

        # Combine prompts with image
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
                "images": [f"data:image/jpeg;base64,{image_data}"]
            }
        ]
        
        # Get response from model
        response = llm.invoke(json.dumps(messages))
        
        # Extract information from response
        try:
            # Attempt to parse probability from response
            prob = extract_probability(response) 
            
            return {
                'probability': prob,
                'findings': response,
                'guidance': 'Please consult a healthcare professional for proper diagnosis. This AI analysis is for informational purposes only.'
            }
        except Exception as parsing_error:
            print(f"Error parsing response: {parsing_error}")
            return {
                'probability': 0.5,  # Default probability
                'findings': str(response),
                'guidance': 'Error processing results. Please consult a healthcare professional.'
            }
            
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        return {
            'probability': 0,
            'findings': f"Error analyzing image: {str(e)}",
            'guidance': 'Technical error occurred. Please try again or consult a medical professional.'
        }

def get_chatbot_response(question):
    try:
        prompt = f"""
        You are a breast cancer awareness chatbot. 
        Only answer questions related to breast cancer.
        
        Question: {question}
        """
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def extract_probability(text):
    """Extract probability from model response text"""
    try:
        # Look for percentage in the text
        import re
        matches = re.findall(r'(\d+)%', text)
        if matches:
            return int(matches[0]) / 100
        return 0.5  # Default probability if no percentage found
    except:
        return 0.5
