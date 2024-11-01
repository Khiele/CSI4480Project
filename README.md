# Email Phishing Detector

A Flask-based web application that uses an ensemble approach combining rule-based analysis and machine learning to detect phishing emails. The system provides detailed analysis and confidence scores for potential phishing attempts.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![CUDA Required](https://img.shields.io/badge/CUDA-required-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## 🚀 Features

- Fine Tuned Llama3.1 llm for phishing email detection
- Web interface for easy email analysis
- Real-time phishing detection
- Detailed feature analysis and scoring
- Machine learning-powered detection
- Rule-based pattern matching
- Comprehensive result visualization
- GPU-accelerated inference

## 📋 Prerequisites

Before you begin, ensure you have the following requirements:

- Python 3.8 or higher
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8 or higher
- 8GB+ GPU memory recommended
- 16GB+ system RAM recommended
- Git
- Visual Studio Code

## 💻 Installation

1. **Unsloth Setup**
- For detailed setup of unsloth in Visual Studio Code, please refer to this tutorial video:
  [Unsloth Setup Guide](https://www.youtube.com/watch?v=UWF6dxQYbU&t=104s)
  - Follow the first 4 steps of the video for proper environment setup
- Additional documentation can be found in the official unsloth documentation:
  [Unsloth Documentation](https://github.com/unslothai/unsloth)

2. **Clone the repository**
```bash
git clone https://github.com/yourusername/phishing-detector.git
cd phishing-detector
```

3. **Install Remaining Python Dependencies**
```bash
pip install flask
```

[Rest of README remains the same as before with sections on:]
- Project Structure
- Configuration
- Running the Application
- Understanding Results
- Security Considerations
- Customization
- Logging
- Troubleshooting
- Performance Optimization
- Contributing
- License
- Contact

## ❗ Important Note
Make sure you have completed the unsloth setup from the video tutorial before proceeding with the rest of the installation. Proper setup of unsloth and CUDA is crucial for the application to function correctly.

## 🐛 Troubleshooting Unsloth

If you encounter issues with unsloth:
1. Verify your CUDA installation matches the requirements specified in the unsloth documentation
2. Ensure your Python environment matches where you are installing dependencies
3. Check the unsloth GitHub issues page for known problems and solutions
4. Make sure your GPU meets the minimum requirements

For additional help with unsloth setup or issues, refer to:
- The official unsloth documentation
- The unsloth GitHub repository issues section
