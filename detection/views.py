from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Report
from .forms import ReportForm
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
import base64
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Ollama
llm = ChatOllama(
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

def extract_json_from_text(text):
    """Extract JSON content from text by finding outermost { and }"""
    try:
        # Convert AIMessage to string if needed
        if hasattr(text, 'content'):
            text = text.content
            
        start_idx = text.find('{')
        end_idx = text.rindex('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            return json.loads(json_str)
    except Exception as e:
        logger.error(f"JSON extraction error: {e}")
    return None

def analyze_image(image_path):
    try:
        # Read and encode image
        with open(image_path, 'rb') as img:
            image_data = base64.b64encode(img.read()).decode('utf-8')
        
        # Step 1: Simple text extraction
        text_extraction_parts = [
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_data}"
            },
            {
                "type": "text",
                "text": "Extract all text from this image."
            }
        ]
        
        message = HumanMessage(content=text_extraction_parts)
        raw_text = llm.invoke([message])
        logger.debug(f"Raw extracted text: {raw_text}")
        
        # Step 2: Analysis with JSON output request
        analysis_parts = [
            {
                "type": "text",
                "text": f"""Based on this medical report text:
                {raw_text.content}
                
                Analyze the severity and provide a response in this exact JSON format, where:
                - probability should be a number between 0-100 indicating cancer likelihood
                - findings should list all key medical observations
                - recommendations should list suggested actions

                Return ONLY the JSON:
                {{
                    "probability": "85",
                    "findings": [
                        "finding 1",
                        "finding 2"
                    ],
                    "recommendations": [
                        "recommendation 1",
                        "recommendation 2"
                    ]
                }}"""
            }
        ]
        
        message = HumanMessage(content=analysis_parts)
        analysis = llm.invoke([message])
        logger.debug(f"Analysis response: {analysis}")
        
        # Extract JSON from response
        json_data = extract_json_from_text(analysis)
        
        if json_data:
            probability = json_data.get('probability')
            # Handle null probability
            if probability is None:
                probability = "50"  # Default value if null
                
            return {
                'probability': int(probability) / 100,
                'findings': '\n'.join(json_data.get('findings', [])),
                'guidance': '\n'.join(json_data.get('recommendations', []))
            }
        else:
            logger.error("Failed to extract JSON from response")
            raise ValueError("Could not extract JSON from response")
            
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        return {
            'probability': 0.5,  # Default probability
            'findings': f"Error analyzing image: {str(e)}",
            'guidance': 'Technical error occurred. Please try again.'
        }

def get_chatbot_response(question):
    try:
        message = HumanMessage(content=[{
            "type": "text",
            "text": f"""You are a breast cancer awareness chatbot.
            Only answer questions related to breast cancer.
            
            Question: {question}"""
        }])
        
        response = llm.invoke([message])
        return output_parser.invoke(response)
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
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
