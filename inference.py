from unsloth import FastLanguageModel
import torch
import logging
import re
import numpy as np

class PhishingDetector:
    def __init__(self, model_path="./phishing_detection_model"):
        """Initialize the phishing detection system"""
        self.model, self.tokenizer = self._load_model(model_path)
        
        # Predefined feature lists
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
        """
        Extract URLs from text using regex
        """
        # Regex to match URLs
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(/[^\s]*)?'
        return re.findall(url_pattern, text)

    def _extract_email_features(self, email_text):
        """
        Extract and score email features for phishing detection
        
        Returns a feature score and detailed feature analysis
        """
        # Lowercase for case-insensitive matching
        lower_text = email_text.lower()
        
        # Initialize feature scores
        feature_analysis = {
            'suspicious_domain_score': 0,
            'urgent_language_score': 0,
            'url_suspicious_score': 0,
            'personal_info_request_score': 0,
            'regex_pattern_score': 0
        }
        
        # URL extraction
        urls = self._extract_urls(email_text)
        
        # Domain suspicious check
        if any(domain in lower_text for domain in self.suspicious_domains):
            feature_analysis['suspicious_domain_score'] = 25
        
        # Urgent language check
        if any(keyword in lower_text for keyword in self.urgent_keywords):
            feature_analysis['urgent_language_score'] = 20
        
        # URL suspicious checks
        for url in urls:
            if any(susp in url.lower() for susp in ['verify', 'secure', 'confirm']):
                feature_analysis['url_suspicious_score'] += 15
            
            # Check URL formatting
            if not re.match(r'^https?://[^/]+\.[^/]+', url):
                feature_analysis['url_suspicious_score'] += 10
        
        # Personal info request detection
        if re.search(r'(confirm|verify|update)\s+(your|account|information)', lower_text, re.IGNORECASE):
            feature_analysis['personal_info_request_score'] = 25
        
        # Regex pattern matching
        for pattern in self.phishing_patterns:
            if re.search(pattern, lower_text, re.IGNORECASE):
                feature_analysis['regex_pattern_score'] += 15
        
        # Calculate total feature score
        total_score = sum(feature_analysis.values())
        
        return total_score, feature_analysis

    def _ml_based_detection(self, email_text):
        """
        Use ML model for phishing detection
        
        Returns classification confidence
        """
        try:
            # Comprehensive prompt for model
            prompt = f"Analyze if this email is a phishing attempt. Provide a confidence score (0-100%).\n\nEmail Content:\n{email_text}"
            
            # Prepare inputs
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            
            # Generate response
            with torch.amp.autocast("cuda"):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract confidence
            confidence_match = re.search(r'(\d+)%', response)
            confidence = int(confidence_match.group(1)) if confidence_match else 50
            
            return confidence
        
        except Exception as e:
            logging.error(f"ML Detection Error: {e}")
            return 50

    def detect_phishing(self, email_text):
        """
        Ensemble approach to phishing detection
        
        Combines rule-based and ML-based methods
        """
        # Feature-based detection
        feature_score, feature_analysis = self._extract_email_features(email_text)
        
        # ML-based detection
        ml_confidence = self._ml_based_detection(email_text)
        
        # Ensemble classification
        def aggregate_results():
            # Weighted combination of feature score and ML confidence
            total_phishing_score = (feature_score + ml_confidence) / 2
            
            # Classification threshold
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

def batch_detect_phishing(emails):
    """Batch phishing detection with comprehensive analysis"""
    detector = PhishingDetector()
    results = []
    
    for i, email in enumerate(emails, 1):
        try:
            logging.info(f"Analyzing Email {i}")
            result = detector.detect_phishing(email)
            result['email'] = email  # Add original email to result
            results.append(result)
        except Exception as e:
            logging.error(f"Error processing email {i}: {e}")
            results.append({
                'email': email,
                'classification': "ERROR",
                'confidence': 0,
                'error': str(e)
            })
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    return results

def main():
    # Comprehensive test suite
    test_emails = [
        # Phishing email examples
        """From: alerts@paypal-secure.verification.com
Subject: Your PayPal Account Has Been Suspended

Dear PayPal User,

Your account has been temporarily suspended due to suspicious activity. 
Immediately verify your identity by clicking this link:
http://totally-not-legit-paypal-verification.com/login""",
        
        # Suspicious email
        """From: support@microsft-update.com
Subject: Critical Windows Security Update Required

Download this urgent security patch immediately:
[Suspicious Link]""",
        
        # Legitimate email
        """From: support@github.com
Subject: Your Monthly GitHub Activity Report

Hi there,
Here's a summary of your recent GitHub contributions...""",
        
        # Complex phishing attempt
        """From: no-reply@yourbanksecure.com
Subject: Urgent: Action Required to Confirm Your Account Information

Dear Customer,

We detected unusual activity on your account ending in *5698* on October 30, 2024. 
For your protection, we have temporarily restricted your online access.

Please verify your account information to lift the restriction:
http://yourbank-verification.com/login"""
    ]
    
    # Run batch detection
    results = batch_detect_phishing(test_emails)
    
    # Structured output
    print("\n=== Phishing Detection Results ===")
    for i, result in enumerate(results, 1):
        print(f"\nEmail {i}:")
        print(f"Original Email: {result['email'][:200]}...")  # Truncate long emails
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']}%")
        
        # Detailed feature analysis
        print("\nFeature Analysis:")
        for feature, score in result['feature_analysis'].items():
            print(f"  {feature}: {score}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()