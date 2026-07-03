import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

import feature_extractor

app = Flask(__name__)

# Paths to the saved model and features list
MODEL_PATH = os.path.join("models", "phishing_detector_model.joblib")
FEATURES_PATH = os.path.join("models", "feature_cols.joblib")

# Global variables for model and features list
clf = None
feature_cols = None

def load_model():
    global clf, feature_cols
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        print("Loading saved model and features list...")
        clf = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURES_PATH)
        print("Model loaded successfully!")
    else:
        print("Warning: Model files not found. Please run train.py first.")

# Ensure the model is loaded at app startup
load_model()

def generate_url_diagnostics(features):
    """
    Analyzes the extracted features and generates human-readable diagnostics.
    Returns a list of dictionaries containing analysis details.
    """
    diagnostics = []
    
    # 1. HTTPS Protocol
    if features['is_https'] == 1:
        diagnostics.append({
            "name": "Connection Protocol",
            "status": "success",
            "value": "HTTPS",
            "description": "Secure connection protocol (HTTPS) is used."
        })
    else:
        diagnostics.append({
            "name": "Connection Protocol",
            "status": "warning",
            "value": "HTTP",
            "description": "Unencrypted HTTP is used. Legitimate login or banking sites almost always require HTTPS."
        })
        
    # 2. IP Address Hostname
    if features['is_ip'] == 1:
        diagnostics.append({
            "name": "IP Hostname",
            "status": "danger",
            "value": "IP Address Detected",
            "description": "The URL uses a raw IP address instead of a domain name, a very common phishing tactic to hide identity."
        })
        
    # 3. Brand Spoofing in Subdomains
    if features['brand_in_subdomain'] == 1:
        diagnostics.append({
            "name": "Brand Misdirection",
            "status": "danger",
            "value": "Brand Spoofing",
            "description": "A well-known brand name is embedded in the subdomain while the actual domain is different (e.g. paypal.someattacker.com)."
        })
        
    # 4. TLD in Path or Subdomain
    if features['tld_in_path'] == 1:
        diagnostics.append({
            "name": "Embedded Domain",
            "status": "danger",
            "value": "TLD in Path",
            "description": "A domain extension (like .com or .org) was found inside the path, indicating attempts to mock a legitimate domain."
        })
    if features['tld_in_subdomain'] == 1:
        diagnostics.append({
            "name": "Embedded Domain",
            "status": "danger",
            "value": "TLD in Subdomain",
            "description": "A domain extension (like .com or .org) was found in the subdomain, which is highly indicative of brand spoofing."
        })

    # 5. URL Shorteners
    if features['is_shortened'] == 1:
        diagnostics.append({
            "name": "URL Shortener",
            "status": "warning",
            "value": "Shortened URL",
            "description": "This URL is obfuscated using a link shortening service. Attackers often use shorteners to hide the actual landing page."
        })
        
    # 6. Suspicious Keywords
    kw_count = features['keyword_count']
    if kw_count > 0:
        diagnostics.append({
            "name": "Sensitive Keywords",
            "status": "warning",
            "value": f"{kw_count} keyword(s)",
            "description": f"URL contains sensitive terms (like 'login', 'secure', 'bank', etc.) often used to deceive users into providing credentials."
        })
        
    # 7. URL Length
    url_len = features['url_length']
    if url_len > 75:
        diagnostics.append({
            "name": "URL Length",
            "status": "warning",
            "value": f"{url_len} characters",
            "description": "The URL is unusually long. Phishing URLs are often very long to conceal the true hostname in mobile browser bars."
        })
    else:
        diagnostics.append({
            "name": "URL Length",
            "status": "success",
            "value": f"{url_len} characters",
            "description": "URL length is within standard, trustworthy limits."
        })

    # 8. Dots in Subdomain / Host
    sub_count = features['subdomain_count']
    if sub_count > 2:
        diagnostics.append({
            "name": "Subdomain Count",
            "status": "warning",
            "value": f"{sub_count} subdomains",
            "description": "The domain has multiple subdomains (e.g. login.secure.verify.site.com), which is common in complex phishing redirects."
        })

    # 9. URL Entropy
    entropy = features['entropy_url']
    if entropy > 4.5:
        diagnostics.append({
            "name": "Character Entropy",
            "status": "warning",
            "value": f"{entropy:.2f} (High)",
            "description": "High character randomness (entropy) detected, which suggests obfuscated scripts, tokens, or auto-generated attack paths."
        })
    else:
        diagnostics.append({
            "name": "Character Entropy",
            "status": "success",
            "value": f"{entropy:.2f} (Normal)",
            "description": "Character distribution is typical of standard URLs."
        })

    # 10. Obfuscated Redirects
    if features['has_double_slash_path'] == 1:
        diagnostics.append({
            "name": "Redirect Pattern",
            "status": "danger",
            "value": "Double Slash '//' in Path",
            "description": "A double slash was found in the URL path. This is a common method to trick web services into redirection."
        })

    return diagnostics

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global clf, feature_cols
    
    # Reload model if it failed to load at startup (or was trained after startup)
    if clf is None or feature_cols is None:
        load_model()
        if clf is None:
            return jsonify({
                "error": "Machine learning model is not available. Please run model training first."
            }), 500
            
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "Missing 'url' parameter in JSON payload."}), 400
            
        url = data['url'].strip()
        if not url:
            return jsonify({"error": "URL cannot be empty."}), 400
            
        # Extract features
        features = feature_extractor.extract_features(url)
        
        # Build pandas DataFrame for prediction (must match train columns exactly)
        df_features = pd.DataFrame([features])
        df_features = df_features[feature_cols] # Align columns
        
        # Get prediction and probabilities
        pred = int(clf.predict(df_features)[0])
        prob = clf.predict_proba(df_features)[0]
        
        # Risk score is the probability of class 1 (phishing/bad)
        risk_score = float(prob[1]) * 100
        
        # Determine status classification
        if risk_score >= 70:
            status = "dangerous"
        elif risk_score >= 35:
            status = "suspicious"
        else:
            status = "safe"
            
        # Generate detailed diagnostics
        diagnostics = generate_url_diagnostics(features)
        
        # Return response
        return jsonify({
            "url": url,
            "prediction": pred,
            "risk_score": round(risk_score, 1),
            "status": status,
            "diagnostics": diagnostics,
            "features": features
        })
        
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # Listen on localhost:5000
    app.run(host='127.0.0.1', port=5000, debug=True)
