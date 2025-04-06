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
git clone https://github.com/SHAKTHI84/Enhanced-AI-Breast-Cancer-Detection.gitgit clone https://github.com/SHAKTHI84/Enhanced-AI-Breast-Cancer-Detection.git
cd Enhanced-AI-Breast-Cancer-Detectioncd Enhanced-AI-Breast-Cancer-Detection
```

2. Create and activate virtual environment:ate virtual environment:
```bash
python -m venv venvon -m venv venv
source venv/bin/activate  # Linux/MacMac
# orr
.\venv\Scripts\activate  # Windows.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bashbash
pip install -r requirements.txtpip install -r requirements.txt
```

4. Set up the database:
```bashbash
python manage.py migratepython manage.py migrate
```

5. Create required directories:ectories:
```bash
mkdir -p media/reportsorts
mkdir -p conversationsir -p conversations
mkdir -p knowledgemkdir -p knowledge
```

6. Start Ollama server:ama server:
```bashbash
ollama serveollama serve
```

7. Run the development server:ver:
```bashbash
python manage.py runserverpython manage.py runserver
```

## 📁 Project Structure📁 Project Structure

```
medisight/
├── core/                 # Project settingsings
├── detection/           # Main applicationon
│   ├── templates/      # HTML templatesates
│   ├── models.py       # Database modelsdels
│   ├── views.py        # View logic
│   └── urls.py         # URL routing
├── media/              # Uploaded fileses
│   └── reports/       # Mammogram images
├── conversations/      # Chat history
├── knowledge/         # Medical reference docs knowledge/         # Medical reference docs
└── manage.py          # Django management└── manage.py          # Django management
```

## 🔍 Usage

1. Access the application at `http://localhost:8000`:8000`
2. Upload mammogram images for analysis
3. Use the chat interface for medical guidance
4. Complete risk assessment questionnaire4. Complete risk assessment questionnaire
5. Review detailed analysis and recommendationsnalysis and recommendations

## 🛠️ Configuration

- **AI Model**: Update Ollama URL in `views.py`
- **Knowledge Base**: Add medical documents to `knowledge/` directory- **Knowledge Base**: Add medical documents to `knowledge/` directory
- **Conversation History**: Stored in `conversations/` directorytory**: Stored in `conversations/` directory

## 🔐 Security Notes

- Development settings are not suitable for productionot suitable for production
- Implement proper authentication before deployment
- Secure the Ollama endpoint- Secure the Ollama endpoint
- Handle medical data according to HIPAA guidelinescal data according to HIPAA guidelines

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.censed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branchranch
3. Commit your changes
4. Push to the branch4. Push to the branch
5. Open a Pull Request5. Open a Pull Request



## 💪 Thanks to all Wonderful Contributors

Thanks a lot for spending your time helping this medisight grow.Thanks a lot for spending your time helping this medisight grow.
Thanks a lot! Keep rocking 🍻Thanks a lot! Keep rocking 🍻



<div align="center"><div align="center">




## License



This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
