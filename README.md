# MediSight: AI-Powered Breast Cancer Detection & Risk Assessment

An intelligent healthcare platform that combines computer vision, natural language processing, and medical expertise to provide breast cancer screening analysis and personalized risk assessment.

## 🌟 Features

- **Advanced Image Analysis**
  - Mammogram processing using Llama Vision
  - Secondary CNN verification
  - Probability-based risk assessment
  - Detailed findings and guidance

- **Interactive Chat System**
  - Context-aware medical assistant
  - Conversation memory
  - Source citations from medical literature
  - Markdown-formatted responses

- **Comprehensive Risk Assessment**
  - 27+ factor analysis
  - Family history evaluation
  - Lifestyle assessment
  - Personalized recommendations
  - Dynamic question flow

## 🔧 Technical Stack

- **Backend**: Django 5.1.7
- **AI Models**: 
  - Llama Vision
  - **Database**: SQLite
- **Frontend**: Bootstrap 5
- **API**: REST-based architecture

## 📋 Prerequisites

- Python 3.13+
- Virtual Environment
- Ollama Server (for AI models)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/SHAKTHI84/SHABreast-Cancer-Detection.git
cd SHABreast-Cancer-Detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the database:
```bash
python manage.py migrate
```

5. Create required directories:
```bash
mkdir -p media/reports
mkdir -p conversations
mkdir -p knowledge
```

6. Start Ollama server:
```bash
ollama serve
```

7. Run the development server:
```bash
python manage.py runserver
```

## 📁 Project Structure

```
medisight/
├── core/                 # Project settings
├── detection/           # Main application
│   ├── templates/      # HTML templates
│   ├── models.py       # Database models
│   ├── views.py        # View logic
│   └── urls.py         # URL routing
├── media/              # Uploaded files
│   └── reports/       # Mammogram images
├── conversations/      # Chat history
├── knowledge/         # Medical reference docs
└── manage.py          # Django management
```

## 🔍 Usage

1. Access the application at `http://localhost:8000`
2. Upload mammogram images for analysis
3. Use the chat interface for medical guidance
4. Complete risk assessment questionnaire
5. Review detailed analysis and recommendations

## 🛠️ Configuration

- **AI Model**: Update Ollama URL in `views.py`
- **Knowledge Base**: Add medical documents to `knowledge/` directory
- **Conversation History**: Stored in `conversations/` directory

## 🔐 Security Notes

- Development settings are not suitable for production
- Implement proper authentication before deployment
- Secure the Ollama endpoint
- Handle medical data according to HIPAA guidelines

## 💪 Thanks to all Wonderful Contributors

Thanks a lot for spending your time helping MediSight grow. 
Keep rocking 🍻

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
