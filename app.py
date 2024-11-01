from flask import Flask, render_template, request, jsonify
from unsloth import FastLanguageModel
import torch
import logging
import re
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PhishingDetector:
    def __init__(self, model_path="./phishing_detection_model"):
        """Initialize the phishing detection system"""
        self.model, self.tokenizer = self._load_model(model_path)
        
        # Predefined feature lists (same as original script)
        self.suspicious_domains = [
            'verification', 'secure', 'bank-confirm', 'account-alert', 
            'urgent-notice', 'verification-team', 'paypal-', 'bank-', 
            'microsoft-', 'apple-', 'support-'
        ]
        
        self.urgent_keywords = [
            'immediately', 'urgent', 'critical', 'suspend', 'restricted', 
            'action required', 'warning', 'alert', 'time sensitive', 
            'your account will be', 'must verify'
        ]
        
        self.phishing_patterns = [
            r'\b(click here|verify now|confirm account)\b',
            r'(suspicious activity|unauthorized transaction)',
            r'(account\s*(will be|has been)\s*(suspended|locked))'
        ]

    def _load_model(self, model_path):
        """Load the fine-tuned model with memory efficiency"""
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True
            )
            
            # Prepare the model for inference
            FastLanguageModel.for_inference(model)
            
            return model, tokenizer
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def _extract_urls(self, text):
        """Extract URLs from text using regex"""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(/[^\s]*)?'
        return re.findall(url_pattern, text)

    def _extract_email_features(self, email_text):
        """Extract and score email features for phishing detection"""
        lower_text = email_text.lower()
        
        feature_analysis = {
            'suspicious_domain_score': 0,
            'urgent_language_score': 0,
            'url_suspicious_score': 0,
            'personal_info_request_score': 0,
            'regex_pattern_score': 0
        }
        
        urls = self._extract_urls(email_text)
        
        if any(domain in lower_text for domain in self.suspicious_domains):
            feature_analysis['suspicious_domain_score'] = 25
        
        if any(keyword in lower_text for keyword in self.urgent_keywords):
            feature_analysis['urgent_language_score'] = 20
        
        for url in urls:
            if any(susp in url.lower() for susp in ['verify', 'secure', 'confirm']):
                feature_analysis['url_suspicious_score'] += 15
            
            if not re.match(r'^https?://[^/]+\.[^/]+', url):
                feature_analysis['url_suspicious_score'] += 10
        
        if re.search(r'(confirm|verify|update)\s+(your|account|information)', lower_text, re.IGNORECASE):
            feature_analysis['personal_info_request_score'] = 25
        
        for pattern in self.phishing_patterns:
            if re.search(pattern, lower_text, re.IGNORECASE):
                feature_analysis['regex_pattern_score'] += 15
        
        total_score = sum(feature_analysis.values())
        
        return total_score, feature_analysis

    def _ml_based_detection(self, email_text):
        """Use ML model for phishing detection"""
        try:
            prompt = f"Analyze if this email is a phishing attempt. Provide a confidence score (0-100%).\n\nEmail Content:\n{email_text}"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.amp.autocast("cuda"):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            confidence_match = re.search(r'(\d+)%', response)
            confidence = int(confidence_match.group(1)) if confidence_match else 50
            
            return confidence
        
        except Exception as e:
            logging.error(f"ML Detection Error: {e}")
            return 50

    def detect_phishing(self, email_text):
        """Ensemble approach to phishing detection"""
        feature_score, feature_analysis = self._extract_email_features(email_text)
        ml_confidence = self._ml_based_detection(email_text)
        
        def aggregate_results():
            total_phishing_score = (feature_score + ml_confidence) / 2
            
            if total_phishing_score > 50:
                classification = "PHISHING"
                confidence = min(total_phishing_score, 100)
            else:
                classification = "SAFE"
                confidence = 100 - total_phishing_score
            
            return {
                'classification': classification,
                'confidence': round(confidence, 2),
                'feature_analysis': feature_analysis
            }
        
        return aggregate_results()

# Flask Application
app = Flask(__name__)

# Global model initialization
detector = PhishingDetector()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Phishing detection endpoint"""
    try:
        # Get email text from request
        email_text = request.form.get('email_text', '').strip()
        
        # Validate input
        if not email_text:
            return jsonify({
                'error': 'No email text provided',
                'status': 'error'
            }), 400
        
        # Detect phishing
        result = detector.detect_phishing(email_text)
        
        # Add extra context based on classification
        if result['classification'] == 'PHISHING':
            result['recommendation'] = "Be cautious! This email shows multiple indicators of being a potential phishing attempt."
        else:
            result['recommendation'] = "This email appears to be safe, but always exercise caution."
        
        return jsonify({
            'result': result,
            'status': 'success'
        })
    
    except Exception as e:
        logging.error(f"Detection error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# Add error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_code=404), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error_code=500), 500

if __name__ == '__main__':
    app.run(debug=True)