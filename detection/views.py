from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.urls import reverse  # Add this import
from django.conf import settings
from .models import Report, RiskAssessment
from .forms import ReportForm, RiskAssessmentForm
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import base64
import json
import logging
import os
from pathlib import Path
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain_core.memory import BaseMemory
from typing import Dict, Any, List
from datetime import datetime
from django.core.exceptions import ValidationError
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Ollama
llm = ChatOllama(
    model="llama3.2-vision",
    base_url="https://9gwg2slc-11434.inc1.devtunnels.ms/",
    temperature=0.1
)

# Configure knowledge base path
KNOWLEDGE_DIR = os.path.join(settings.BASE_DIR, 'knowledge')

# Add these constants at the top
MAX_QUESTIONS = 20
QUESTION_WEIGHTS = {
    'name': 0,  # Doesn't count towards limit
    'age_range': 2,
    'family_history': 3,
    'family_history_age': 2,
    'previous_biopsies': 2,
    'biopsy_results': 2,
    'first_period_age': 1,
    'first_pregnancy_age': 1,
    'hormone_therapy': 1,
    'lifestyle_factors': 2,
    'exercise_frequency': 1,
    'smoking_status': 1,
    'alcohol_consumption': 1,
    'weight_status': 1
}

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
            
            # Store analysis in session for chat context
            image_analyses = request.session.get('image_analyses', [])
            image_analyses.append({
                'image_id': report.id,
                'timestamp': str(report.created_at),
                'probability': report.probability,
                'findings': report.findings,
                'guidance': report.guidance
            })
            request.session['image_analyses'] = image_analyses
            request.session['last_image_id'] = report.id
            
            # Add to chat history
            session_id = request.session.session_key or 'default'
            save_conversation(session_id, "System", 
                f"[Image uploaded and analyzed]\nFindings: {report.findings}\nRisk probability: {report.probability * 100:.1f}%\nGuidance: {report.guidance}")
            
            return redirect('report_detail', pk=report.pk)
    else:
        form = ReportForm()
    return render(request, 'detection/upload.html', {'form': form})

def load_knowledge_base():
    """Load knowledge base documents from the knowledge directory"""
    documents = []
    
    try:
        # Load PDF documents
        pdf_loader = PyPDFDirectoryLoader(KNOWLEDGE_DIR)
        pdf_documents = pdf_loader.load()
        documents.extend(pdf_documents)
        logger.debug(f"Loaded {len(pdf_documents)} PDF documents")

        # Load text documents if any
        txt_loader = DirectoryLoader(KNOWLEDGE_DIR, glob="**/*.txt")
        txt_documents = txt_loader.load()
        documents.extend(txt_documents)
        logger.debug(f"Loaded {len(txt_documents)} text documents")

        return documents
        
    except Exception as e:
        logger.error(f"Error in load_knowledge_base: {e}")
        return []

def get_relevant_context(question, documents, max_docs=2):
    """Find relevant documents for the question"""
    try:
        relevant_docs = []
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            # Simple relevance check - can be improved
            if any(word.lower() in content.lower() for word in question.split()):
                source = metadata.get('source', 'Unknown')
                # Get just the filename from the path
                source = os.path.basename(source)
                relevant_docs.append({
                    'content': content,
                    'source': source,
                    'page': metadata.get('page', 1)
                })
                
        return relevant_docs[:max_docs]  # Return top N most relevant docs
    except Exception as e:
        logger.error(f"Error in get_relevant_context: {e}")
        return []

def chat_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get('question')
            session_id = request.session.session_key or 'default'
            
            # Ensure conversations directory exists
            conversation_dir = os.path.join(settings.BASE_DIR, 'conversations')
            os.makedirs(conversation_dir, exist_ok=True)
            
            # Load ALL conversation files
            all_conversations = []
            for filename in os.listdir(conversation_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(conversation_dir, filename), 'r') as f:
                        file_content = f.readlines()
                        # Add source identifier to each line
                        conversation_id = filename.replace('.txt', '')
                        all_conversations.extend([
                            f"{line.strip()}" for line in file_content
                        ])

            # Get recent conversation history (last 10 exchanges)
            recent_conversations = all_conversations[-10:] if all_conversations else []
            
            # Get knowledge base context
            documents = load_knowledge_base()
            relevant_docs = get_relevant_context(question, documents)
            
            # Format context with ALL conversation history
            context = f"""Previous conversations:
            {chr(10).join(recent_conversations)}
            
            Current question: {question}
            
            Guidelines:
            - Use the conversation history to maintain context
            - Reference previous discussions when relevant
            - Cite sources when providing medical information
            - Use markdown formatting for clear presentation
            - Be compassionate and clear in explanations
            - If contradicting previous statements, explain why
            """
            
            # Create system message
            system_message = HumanMessage(content=[{
                "type": "text",
                "text": context
            }])
            
            # Create user question message
            user_message = HumanMessage(content=[{
                "type": "text",
                "text": question
            }])
            
            # Get response from LLM
            response = llm.invoke([system_message, user_message])
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Save to current session's conversation file
            current_file = os.path.join(conversation_dir, f'{session_id}.txt')
            with open(current_file, 'a') as f:
                f.write(f"User: {question}\n")
                f.write(f"Assistant: {response_text}\n")
            
            # Format response with sources
            if relevant_docs:
                response_text += "\n\n**Sources:**"
                for doc in relevant_docs:
                    response_text += f"\n- {doc['source']} (Page {doc['page']})"
            
            return JsonResponse({
                'response': response_text,
                'references': relevant_docs
            })
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return JsonResponse({
                'response': 'An error occurred while processing your request'
            }, status=500)
    
    # For GET requests, load ALL conversation history
    conversation_history = []
    conversation_dir = os.path.join(settings.BASE_DIR, 'conversations')
    
    if os.path.exists(conversation_dir):
        for filename in os.listdir(conversation_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(conversation_dir, filename), 'r') as f:
                    conversation_history.extend(f.readlines())
    
    return render(request, 'detection/chat.html', {
        'conversation_history': conversation_history
    })

def extract_json_from_text(text):
    """Extract JSON content from text more robustly"""
    try:
        # Convert AIMessage to string if needed
        if hasattr(text, 'content'):
            text = text.content
            
        # First try to find JSON between triple backticks
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
            
        # Then try to find between curly braces
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if (json_match):
            return json.loads(json_match.group(0))
            
        # If still not found, try to parse the whole text
        return json.loads(text)
        
    except Exception as e:
        logger.error(f"JSON extraction error: {str(e)}")
        logger.debug(f"Failed text: {text}")
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
        
        json_data = extract_json_from_text(analysis)
        
        if json_data:
            probability = json_data.get('probability')
            if probability is None:
                probability = "50"
                
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
            'probability': 0.5,
            'findings': f"Error analyzing image: {str(e)}",
            'guidance': 'Technical error occurred. Please try again.'
        }

def get_chatbot_response(question):
    try:
        system_message = HumanMessage(content=[{
            "type": "text",
            "text": """You are a specialized breast cancer awareness chatbot.
            Format your responses using markdown:
            - Use # for main headings
            - Use ## for subheadings
            - Use * for bullet points
            - Use > for important quotes/notes
            - Use `code` for medical terms
            - Use tables when comparing options
            
            Include these topics:
            - Early detection and screening
            - Signs and symptoms
            - Risk factors and prevention
            - Treatment options
            - Support resources
            
            Keep responses well-structured and informative."""
        }])
        
        # Create user question message
        user_message = HumanMessage(content=[{
            "type": "text",
            "text": question
        }])
        
        # Get response from LLM
        response = llm.invoke([system_message, user_message])
        
        # Extract content from AIMessage
        if hasattr(response, 'content'):
            return response.content
        return str(response)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return "I apologize, but I encountered an error. Please try asking your question again."

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

def calculate_risk_score(data):
    """Calculate risk score based on assessment data"""
    score = 0
    
    # Age factor
    age_scores = {
        '20-29': 1, '30-39': 2, '40-49': 3,
        '50-59': 4, '60+': 5
    }
    score += age_scores.get(data['age_range'], 0)
    
    # Family history
    if data['family_history']:
        score += 3
        if data.get('family_history_age', 0) and data['family_history_age'] < 50:
            score += 2
    
    # Personal history
    if data['previous_biopsies']:
        score += 2
        if data.get('biopsy_results'):
            score += 3
    
    # Risk factors
    if data['first_period_age'] < 12:
        score += 2
    
    if data.get('hormone_therapy'):
        score += 1
    
    if data.get('first_pregnancy_age', 0) > 30 or not data.get('first_pregnancy_age'):
        score += 2
    
    # Normalize score to 0-1 range
    max_possible_score = 20
    normalized_score = score / max_possible_score
    
    return normalized_score

def get_recommendations(risk_score):
    """Generate recommendations based on risk score"""
    recommendations = []
    
    if risk_score < 0.3:
        recommendations.extend([
            "Continue with regular breast self-examinations",
            "Schedule annual clinical breast exams",
            "Maintain a healthy lifestyle with regular exercise"
        ])
    elif risk_score < 0.6:
        recommendations.extend([
            "Schedule mammogram screening every 1-2 years",
            "Consider more frequent clinical breast exams",
            "Discuss risk-reduction strategies with your healthcare provider",
            "Review your family history with a genetic counselor"
        ])
    else:
        recommendations.extend([
            "Schedule immediate consultation with a breast specialist",
            "Consider genetic testing for BRCA1/BRCA2 mutations",
            "Discuss preventive measures with your healthcare provider",
            "Consider more frequent screening and monitoring",
            "Review lifestyle factors that might affect risk"
        ])
    
    return recommendations

def get_next_question(current_answers):
    """Get next question based on previous answers"""
    try:
        context = "\n".join([
            f"Q: {q}\nA: {a}" for q, a in current_answers.items()
        ])
        
        system_message = HumanMessage(content=[{
            "type": "text",
            "text": f"""Based on these previous answers:
            {context}
            
            Provide the next question for breast cancer risk assessment.
            Return in JSON format:
            {{
                "message": "your question here",
                "field": "unique_field_name",
                "type": "text or choice",
                "options": ["option1", "option2"] // only for choice type
            }}"""
        }])
        
        response = llm.invoke([system_message])
        return extract_json_from_text(response.content)
    except Exception as e:
        logger.error(f"Error getting next question: {e}")
        return None

def get_initial_question():
    """Get the welcoming question from LLM"""
    system_message = HumanMessage(content=[{
        "type": "text",
        "text": """You are a compassionate medical professional conducting a breast cancer risk assessment.
        Provide a warm welcome and the first question about basic information.
        Use markdown for formatting.
        
        Return response in this JSON format:
        {
            "message": "welcome message and first question",
            "field": "age_range",
            "options": ["20-29", "30-39", "40-49", "50-59", "60+"],
            "type": "choice"
        }"""
    }])
    
    try:
        response = llm.invoke([system_message])
        return extract_json_from_text(response.content)
    except Exception as e:
        logger.error(f"Error getting initial question: {e}")
        # Fallback default question
        return {
            "message": "# Welcome to Your Breast Cancer Risk Assessment\n\nLet's start with a simple question: What is your age range?",
            "field": "age_range",
            "options": ["20-29", "30-39", "40-49", "50-59", "60+"],
            "type": "choice"
        }

class CustomConversationMemory(BaseMemory):
    """Custom memory implementation for risk assessment"""
    
    def __init__(self):
        super().__init__()
        self._messages = []  # Use private attribute
        self._human_prefix = "Human"
        self._ai_prefix = "Assistant"

    @property
    def messages(self):
        """Messages property getter"""
        return self._messages

    def save_context(self, query: str, response: str) -> None:
        """Save the context of a conversation turn"""
        if not hasattr(self, '_messages'):
            self._messages = []
            
        self._messages.append({
            "type": "human",
            "content": str(query)
        })
        self._messages.append({
            "type": "ai", 
            "content": str(response)
        })

    def load_memory_variables(self) -> Dict[str, Any]:
        """Return the stored conversation history"""
        if not hasattr(self, '_messages'):
            return {"history": []}
            
        formatted_history = []
        for msg in self._messages:
            prefix = self._human_prefix if msg["type"] == "human" else self._ai_prefix
            formatted_history.append(f"{prefix}: {msg['content']}")
        
        return {
            "history": formatted_history
        }

    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables required"""
        return ["history"]

    def clear(self) -> None:
        """Clear conversation history"""
        self._messages = []

def save_conversation(session_id: str, role: str, message: str):
    """Save conversation to file with error handling"""
    try:
        filepath = f"conversations/{session_id}.txt"
        os.makedirs("conversations", exist_ok=True)
        
        with open(filepath, "a") as f:
            f.write(f"{role}: {message}\n")
            
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")
        # Continue execution even if save fails

def get_llm_response(context=None, session_id=None):
    """Get next question based on conversation context with strict topic tracking"""
    try:
        # Check if context is provided
        if not context:
            context = {}
            
        # Get user's name from context
        user_name = context.get('name', 'there')
        
        # Track which fields have been asked to prevent repetition
        asked_fields = set(context.keys())
        
        # Define topic categories and map fields to them with status tracking
        topic_categories = {
            'basic_info': {
                'fields': ['name', 'age', 'gender'],
                'covered': any(field in asked_fields for field in ['name', 'age', 'gender']),
                'next_question': {
                    'message': f"# Basic Information\n\n{user_name}, what is your age?",
                    'field': 'age',
                    'type': 'text'
                }
            },
            'family_history': {
                'fields': ['family_history', 'family_history_details', 'family_cancer_age'],
                'covered': any(field in asked_fields for field in ['family_history', 'family_history_details', 'family_cancer_age']),
                'next_question': {
                    'message': f"# Family History\n\n{user_name}, do any of your close relatives (mother, sister, aunt) have a history of breast cancer?",
                    'field': 'family_history_details',
                    'type': 'text'
                }
            },
            'personal_health': {
                'fields': ['personal_health', 'medical_conditions', 'previous_conditions'],
                'covered': any(field in asked_fields for field in ['personal_health', 'medical_conditions', 'previous_conditions']),
                'next_question': {
                    'message': f"# Health History\n\n{user_name}, do you have any ongoing medical conditions or health concerns?",
                    'field': 'medical_conditions',
                    'type': 'text'
                }
            },
            'symptoms': {
                'fields': ['breast_symptoms', 'breast_changes', 'breast_pain', 'breast_tenderness'],
                'covered': any(field in asked_fields for field in ['breast_symptoms', 'breast_changes', 'breast_pain', 'breast_tenderness']),
                'next_question': {
                    'message': f"# Breast Health\n\n{user_name}, have you experienced any unusual symptoms in your breasts like lumps, pain, discharge, or changes in appearance?",
                    'field': 'breast_symptoms',
                    'type': 'text'
                }
            },
            'lifestyle': {
                'fields': ['diet', 'exercise', 'smoking', 'alcohol', 'weight_changes'],
                'covered': any(field in asked_fields for field in ['diet', 'exercise', 'smoking', 'alcohol', 'weight_changes']),
                'next_question': {
                    'message': f"# Lifestyle Factors\n\n{user_name}, how would you describe your lifestyle in terms of diet, exercise, and alcohol consumption?",
                    'field': 'lifestyle_factors',
                    'type': 'text'
                }
            },
            'hormonal': {
                'fields': ['hormone_therapy', 'birth_control', 'hormone_medications'],
                'covered': any(field in asked_fields for field in ['hormone_therapy', 'birth_control', 'hormone_medications']),
                'next_question': {
                    'message': f"# Hormone Treatments\n\n{user_name}, have you ever used hormone replacement therapy or other hormonal treatments?",
                    'field': 'hormone_treatments',
                    'type': 'text'
                }
            },
            'environmental': {
                'fields': ['radiation_exposure', 'chemical_exposure'],
                'covered': any(field in asked_fields for field in ['radiation_exposure', 'chemical_exposure']),
                'next_question': {
                    'message': f"# Environmental Factors\n\n{user_name}, have you been exposed to radiation or harmful chemicals in your work or living environment?",
                    'field': 'environmental_exposure',
                    'type': 'text'
                }
            },
            'reproductive': {
                'fields': ['pregnancy_history', 'menstrual_history', 'menopause_status', 'pcos', 'reproductive_health'],
                'covered': any(field in asked_fields for field in ['pregnancy_history', 'menstrual_history', 'menopause_status', 'pcos', 'reproductive_health']),
                'next_question': {
                    'message': f"# Reproductive Health\n\n{user_name}, could you tell me about any reproductive health factors like pregnancy history, birth control use, or menstrual issues?",
                    'field': 'reproductive_health',
                    'type': 'text'
                }
            }
        }
        
        # Count covered topics and get available topics
        covered_topics = [topic for topic, data in topic_categories.items() if data['covered']]
        available_topics = [topic for topic, data in topic_categories.items() if not data['covered']]
        
        # Format previous answers for logging
        previous_answers = []
        for field, answer in context.items():
            if field == 'name':
                previous_answers.append(f"Name: {answer}")
            else:
                readable_field = field.replace('_', ' ').title()
                previous_answers.append(f"Q: {readable_field}\nA: {answer}")
        
        # Complete if we've covered most topics or asked enough questions
        if len(covered_topics) >= 5 or len(context) >= 10:
            return {
                'completed': True,
                'message': f"# Assessment Complete\n\nThank you {user_name} for providing all this information. Let me analyze your responses and provide you with a personalized risk assessment."
            }
        
        # Check if there are any topics left
        if not available_topics:
            return {
                'completed': True,
                'message': f"# Assessment Complete\n\nThank you {user_name} for answering all my questions. I'll now provide you with a personalized risk assessment."
            }
            
        # Get next topic and question
        next_topic = available_topics[0]
        next_question = topic_categories[next_topic]['next_question']
        
        return next_question

    except Exception as e:
        logger.error(f"LLM Error: {str(e)}")
        return {
            "message": f"# One Last Question\n\n{user_name}, is there anything else you'd like to share about your health that might be relevant for this assessment?",
            "type": "text",
            "field": "additional_information"
        }

def analyze_conversation(session_id: str, context: dict = None) -> Dict:
    """Analyze conversation for risk assessment using both file and context"""
    try:
        # Format context information
        context_items = []
        if context:
            for field, answer in context.items():
                if field == 'name':
                    name = answer
                    context_items.append(f"Patient name: {answer}")
                else:
                    readable_field = field.replace('_', ' ').title()
                    context_items.append(f"Q: Tell me about your {readable_field}")
                    context_items.append(f"A: {answer}")
        
        # Get name from context
        name = context.get('name', 'Anonymous') if context else 'Anonymous'
        
        # Also read conversation file for backup
        conversation_file = f"conversations/{session_id}.txt"
        file_content = ""
        if os.path.exists(conversation_file):
            with open(conversation_file, 'r') as f:
                file_content = f.read()
        
        # Combine both sources
        conversation = "\n".join(context_items) + "\n\n" + file_content
        
        analysis_prompt = HumanMessage(content=[{
            "type": "text",
            "text": f"""Analyze this breast cancer risk assessment conversation in detail:

{conversation}

Consider all factors mentioned including:
- Age and family history of breast cancer
- Medical history and conditions 
- Lifestyle factors (exercise, diet, smoking, alcohol)
- Reproductive history (age at first period, pregnancies, hormone use)
- Current symptoms or concerns
- Environmental exposures

For {name}, provide a comprehensive analysis in JSON format:
{{
    "name": "{name}",
    "risk_level": "LOW, MODERATE, or HIGH",
    "risk_score": "0-1 floating point (where 0 is lowest risk, 1 is highest)",
    "key_factors": ["list all risk factors identified in the conversation"],
    "recommendations": ["personalized recommendations based on their specific situation"],
    "summary": "detailed analysis explanation with compassionate tone"
}}"""
        }])

        response = llm.invoke([analysis_prompt])
        result = extract_json_from_text(response.content)
        
        # Ensure name is included
        if result and not result.get('name'):
            result['name'] = name
        
        return result

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {
            "name": name if 'name' in locals() else "Anonymous",
            "risk_level": "MODERATE", 
            "risk_score": "0.5",
            "key_factors": ["Unable to analyze conversation completely"],
            "recommendations": ["Please consult with a healthcare provider for a complete assessment"],
            "summary": "An error occurred while analyzing your responses."
        }

# Define a fixed sequence of questions for the risk assessment
RISK_ASSESSMENT_QUESTIONS = [
    {
        "message": "# Welcome to Your Breast Cancer Risk Assessment\n\nI'm here to understand your personal risk factors. To start, what's your name?",
        "field": "name",
        "type": "text"
    },
    {
        "message": "# Basic Information\n\nWhat is your age range?",
        "field": "age_range",
        "type": "choice",
        "options": ["18-29", "30-39", "40-49", "50-59", "60+"]
    },
    {
        "message": "# Family History\n\nDo you have a family history of breast cancer (mother, sister, daughter, etc.)?",
        "field": "family_history",
        "type": "choice",
        "options": ["Yes", "No", "Not sure"]
    },
    {
        "message": "# Family History Details\n\nIf you answered yes to family history, which family members had breast cancer and at what ages were they diagnosed?",
        "field": "family_history_details",
        "type": "text"
    },
    {
        "message": "# Medical History\n\nHave you ever had a breast biopsy?",
        "field": "previous_biopsies",
        "type": "choice",
        "options": ["Yes", "No"]
    },
    {
        "message": "# Breast Health\n\nHave you noticed any unusual symptoms or changes in your breasts recently?",
        "field": "breast_symptoms",
        "type": "text"
    },
    {
        "message": "# Reproductive History\n\nAt what age did you have your first menstrual period?",
        "field": "first_period_age",
        "type": "text"
    },
    {
        "message": "# Reproductive Health\n\nHave you ever been diagnosed with any reproductive conditions like PCOS?",
        "field": "reproductive_conditions",
        "type": "text"
    },
    {
        "message": "# Pregnancy History\n\nHave you ever been pregnant? If so, at what age was your first pregnancy?",
        "field": "pregnancy_history",
        "type": "text"
    },
    {
        "message": "# Menstrual Health\n\nDo you experience any menstrual irregularities?",
        "field": "menstrual_irregularities",
        "type": "text"
    },
    {
        "message": "# Hormone Use\n\nHave you ever used hormone replacement therapy or hormonal contraceptives?",
        "field": "hormone_use",
        "type": "text"
    },
    {
        "message": "# Lifestyle\n\nHow would you describe your diet and exercise habits?",
        "field": "lifestyle_habits",
        "type": "text"
    },
    {
        "message": "# Alcohol Consumption\n\nHow often do you consume alcoholic beverages?",
        "field": "alcohol_consumption",
        "type": "choice",
        "options": ["Never", "Occasionally", "Weekly", "Daily"]
    },
    {
        "message": "# Smoking Status\n\nDo you smoke or have you smoked in the past?",
        "field": "smoking_status",
        "type": "choice",
        "options": ["Never smoked", "Former smoker", "Current smoker"]
    },
    {
        "message": "# Body Weight\n\nHave you experienced any significant weight changes in the past few years?",
        "field": "weight_changes",
        "type": "text"
    },
    {
        "message": "# Environmental Factors\n\nHave you been exposed to radiation treatments or environmental toxins?",
        "field": "environmental_exposure",
        "type": "text"
    },
    {
        "message": "# Breast Density\n\nIf you've had a mammogram, were you told you have dense breast tissue?",
        "field": "breast_density",
        "type": "choice",
        "options": ["Yes", "No", "Not sure", "Never had a mammogram"]
    },
    {
        "message": "# Medical Conditions\n\nDo you have any other medical conditions we should be aware of?",
        "field": "medical_conditions",
        "type": "text"
    },
    {
        "message": "# Medications\n\nAre you currently taking any medications regularly?",
        "field": "medications",
        "type": "text"
    },
    {
        "message": "# Physical Activity\n\nHow often do you exercise or engage in physical activities?",
        "field": "physical_activity",
        "type": "choice",
        "options": ["Rarely", "1-2 times/week", "3-4 times/week", "5+ times/week"]
    },
    {
        "message": "# Stress Levels\n\nHow would you describe your typical stress levels?",
        "field": "stress_levels",
        "type": "choice",
        "options": ["Low", "Moderate", "High", "Very high"]
    },
    {
        "message": "# Sleep Habits\n\nOn average, how many hours of sleep do you get per night?",
        "field": "sleep_habits",
        "type": "text"
    },
    {
        "message": "# Genetic Testing\n\nHave you ever had genetic testing for breast cancer risk (like BRCA1/BRCA2)?",
        "field": "genetic_testing",
        "type": "choice",
        "options": ["Yes", "No", "Not sure"]
    },
    {
        "message": "# Breast Self-Exams\n\nDo you perform regular breast self-examinations?",
        "field": "breast_self_exams",
        "type": "choice",
        "options": ["Yes, monthly", "Yes, occasionally", "No"]
    },
    {
        "message": "# Recent Mammogram\n\nWhen was your last mammogram or breast screening?",
        "field": "last_mammogram",
        "type": "text"
    },
    {
        "message": "# Breast Pain\n\nDo you experience regular breast pain or tenderness?",
        "field": "breast_pain",
        "type": "text"
    },
    {
        "message": "# Childhood Health\n\nDid you experience any significant health issues during childhood?",
        "field": "childhood_health",
        "type": "text"
    },
    {
        "message": "# Additional Information\n\nIs there anything else you'd like to share about your health that might be relevant to this assessment?",
        "field": "additional_information",
        "type": "text"
    }
]

def risk_assessment(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = request.session.session_key or 'default'
            
            # Initialize session if needed
            if 'risk_context' not in request.session:
                request.session['risk_context'] = {}
            if 'question_index' not in request.session:
                request.session['question_index'] = -1  # Start at -1
                
            context = request.session.get('risk_context', {})
            question_index = request.session.get('question_index', -1)

            # Handle start of assessment
            if data.get('start'):
                context = {}
                question_index = 0
                request.session['risk_context'] = context
                request.session['question_index'] = question_index
                return JsonResponse(RISK_ASSESSMENT_QUESTIONS[0])

            # Handle question responses
            if 'field' in data and 'answer' in data:
                field = data['field']
                answer = data['answer']
                
                # Save answer
                context[field] = answer
                request.session['risk_context'] = context
                
                # Save to conversation history
                save_conversation(session_id, "User", f"{field}: {answer}")
                
                # Move to next question
                question_index += 1
                request.session['question_index'] = question_index
                
                # Check if assessment is complete
                if question_index >= len(RISK_ASSESSMENT_QUESTIONS):
                    try:
                        analysis = analyze_conversation(session_id, context)
                        if not analysis:
                            raise ValueError("Analysis failed to return results")
                            
                        assessment = RiskAssessment.objects.create(
                            name=context.get('name', 'Anonymous'),
                            conversation_history=json.dumps(analysis),
                            risk_level=analysis.get('risk_level', 'MODERATE'),
                            risk_score=float(analysis.get('risk_score', 0.5)),
                            key_factors='\n'.join(analysis.get('key_factors', [])),
                            recommendations='\n'.join(analysis.get('recommendations', [])),
                            summary=analysis.get('summary', 'Analysis completed')
                        )
                        
                        # Clear session
                        request.session['risk_context'] = {}
                        request.session['question_index'] = -1
                        
                        return JsonResponse({
                            'completed': True,
                            'redirect_url': reverse('risk_result', kwargs={'pk': assessment.pk})
                        })
                    except Exception as e:
                        logger.error(f"Error creating assessment: {str(e)}")
                        raise
                
                try:
                    # Get next question
                    next_question = RISK_ASSESSMENT_QUESTIONS[question_index]
                    
                    # Handle conditional questions
                    if next_question['field'] == 'family_history_details':
                        if context.get('family_history', '').lower() == 'no':
                            question_index += 1
                            request.session['question_index'] = question_index
                            next_question = RISK_ASSESSMENT_QUESTIONS[question_index]
                    
                    # Format question with name if available
                    if 'name' in context and '{name}' in next_question.get('message', ''):
                        next_question['message'] = next_question['message'].format(
                            name=context['name']
                        )
                    
                    # Save question to history
                    save_conversation(session_id, "Assistant", next_question['message'])
                    
                    return JsonResponse(next_question)
                    
                except IndexError:
                    logger.error(f"Question index out of range: {question_index}")
                    return JsonResponse({
                        'error': 'Assessment structure error',
                        'message': 'Unable to load next question'
                    }, status=500)

        except Exception as e:
            logger.error(f"Risk assessment error: {str(e)}")
            return JsonResponse({
                'error': str(e),
                'message': 'An error occurred processing your response'
            }, status=500)

    # GET request - render initial page
    return render(request, 'detection/risk_assessment.html')

def risk_result(request, pk):
    """Display risk assessment results"""
    assessment = RiskAssessment.objects.get(pk=pk)
    
    # Convert risk score to percentage
    risk_percentage = int(assessment.risk_score * 100)
    
    # Get recommendations as list
    recommendations = assessment.recommendations.split('\n')
    
    context = {
        'assessment': assessment,
        'risk_percentage': risk_percentage,
        'recommendations': recommendations,
    }
    
    return render(request, 'detection/risk_result.html', {'context': context})

def create_risk_assessment(data: Dict[str, Any]) -> RiskAssessment:
    """Create a risk assessment record and analyze responses"""
    try:
        # Format conversation history for analysis
        conversation = []
        for field, answer in data.items():
            if field == 'name':
                conversation.append(f"Patient Name: {answer}")
            else:
                conversation.append(f"Q: Tell me about your {field.replace('_', ' ')}")
                conversation.append(f"A: {answer}")

        # Ask LLM to analyze the conversation
        analysis_prompt = HumanMessage(content=[{
            "type": "text",
            "text": f"""Analyze this breast cancer risk assessment conversation:

{chr(10).join(conversation)}

Provide a risk analysis in JSON format:
{{
    "risk_level": "LOW, MODERATE, or HIGH",
    "risk_score": "0-1 floating point number",
    "key_factors": ["list of identified risk factors"],
    "recommendations": ["list of recommendations"],
    "summary": "brief analysis explanation"
}}"""
        }])

        response = llm.invoke([analysis_prompt])
        analysis = extract_json_from_text(response.content)

        if not analysis:
            raise ValueError("Could not analyze responses")

        # Create assessment record
        assessment = RiskAssessment.objects.create(
            name=data.get('name', 'Anonymous'),
            conversation_history=json.dumps(conversation),
            risk_level=analysis['risk_level'],
            risk_score=float(analysis['risk_score']),
            key_factors='\n'.join(analysis['key_factors']),
            recommendations='\n'.join(analysis['recommendations']),
            summary=analysis['summary']
        )

        return assessment

    except Exception as e:
        logger.error(f"Error creating risk assessment: {e}")
        raise

def report_detail(request, pk):
    """Display report details with chat interface"""
    report = Report.objects.get(pk=pk)
    
    # Calculate probability percentage for template
    probability_percentage = round(report.probability * 100, 1)
    
    # Store current report as last viewed in session
    request.session['last_image_id'] = report.pk
    
    # Get conversation history related to this image
    conversation_history = []
    session_id = request.session.session_key or 'default'
    conversation_file = f"conversations/{session_id}.txt"
    if os.path.exists(conversation_file):
        with open(conversation_file, 'r') as f:
            conversation_history = [line for line in f.readlines() 
                                  if f"Image {report.pk}" in line or "findings" in line.lower()]
    
    # Get initial explanation from LLM if no conversation history
    if not conversation_history:
        explanation_prompt = HumanMessage(content=[{
            "type": "text",
            "text": f"""You are a medical AI assistant explaining a breast cancer screening result.
            
            Image analysis results:
            - Probability: {probability_percentage}%
            - Findings: {report.findings}
            - Guidance: {report.guidance}
            
            Provide a compassionate initial explanation of these results in a conversational tone.
            Use markdown formatting and speak directly to the patient.
            Be honest but reassuring, and explain what these results mean.
            """
        }])
        
        explanation = llm.invoke([explanation_prompt])
        initial_explanation = explanation.content if hasattr(explanation, 'content') else str(explanation)
        
        # Save this explanation to conversation history
        save_conversation(session_id, "System", f"[Image {report.pk} uploaded for analysis]")
        save_conversation(session_id, "Assistant", initial_explanation)
    else:
        initial_explanation = None
    
    return render(request, 'detection/detail.html', {
        'report': report,
        'probability_percentage': probability_percentage,
        'initial_explanation': initial_explanation,
        'conversation_history': conversation_history
    })

def analyze_image_chat(request):
    """Handle chat about a specific image"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get('question')
            image_id = data.get('image_id')
            
            if not image_id:
                return JsonResponse({'error': 'No image specified'}, status=400)
                
            # Get the report
            report = Report.objects.get(pk=image_id)
            
            # Create context with image details
            image_context = f"""
            Breast cancer screening image analysis results:
            - Risk probability: {report.probability * 100:.1f}%
            - Findings: {report.findings}
            - Recommended guidance: {report.guidance}
            """
            
            session_id = request.session.session_key or 'default'
            
            # Load conversation history
            conversation_history = []
            conversation_file = f"conversations/{session_id}.txt"
            if os.path.exists(conversation_file):
                with open(conversation_file, 'r') as f:
                    conversation_history = [line for line in f.readlines() 
                                          if f"Image {image_id}" in line 
                                          or "findings" in line.lower()]
            
            # Create system message with image context
            system_message = HumanMessage(content=[{
                "type": "text",
                "text": f"""You are a medical AI assistant discussing a specific breast cancer screening result.
                
                {image_context}
                
                Previous conversation about this image:
                {' '.join(conversation_history[-10:] if len(conversation_history) > 10 else conversation_history)}
                
                Answer the user's question specifically about these image analysis results.
                Use markdown formatting in your response.
                Be compassionate but accurate in your explanation.
                """
            }])
            
            # Create user question message
            user_message = HumanMessage(content=[{
                "type": "text",
                "text": question
            }])
            
            # Save user question to conversation history
            save_conversation(session_id, "User", f"[About Image {image_id}]: {question}")
            
            # Get response from LLM
            response = llm.invoke([system_message, user_message])
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Save AI response to conversation history
            save_conversation(session_id, "Assistant", response_text)
            
            return JsonResponse({
                'response': response_text,
                'image_id': image_id
            })
            
        except Exception as e:
            logger.error(f"Image chat error: {str(e)}")
            return JsonResponse({
                'response': 'An error occurred while processing your request'
            }, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def clear_chat(request):
    """Clear chat history for current session"""
    if request.method == 'POST':
        session_id = request.session.session_key or 'default'
        
        # Clear session memory
        request.session['chat_memory'] = []
        
        # Optionally archive conversation file instead of deleting
        conversation_file = f"conversations/{session_id}.txt"
        if os.path.exists(conversation_file):
            archive_dir = "conversations/archived"
            os.makedirs(archive_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_file = f"{archive_dir}/{session_id}_{timestamp}.txt"
            os.rename(conversation_file, archived_file)
        
        return JsonResponse({'status': 'success'})
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
