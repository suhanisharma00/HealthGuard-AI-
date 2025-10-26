
<div align="center">
  <h1 style="background-color:#ff4d4d;color:white;padding:10px;border-radius:8px;">🚑 HealthGuard AI</h1>
</div>

*Your personal AI-powered health sidekick — chat, upload reports, and predict diabetes risk.*

![HealthGuard AI Banner](assets/hero.gif)

---

## 🚀 Overview

HealthGuard AI is a Streamlit-based health assistant that combines:

* **Machine Learning (Random Forest)** for diabetes risk prediction.
* **LLM-powered chat** (Google Gemini + Hugging Face fallback) for health Q&A.
* **PDF parsing** to extract and analyze medical reports.
* **Downloadable health reports** with chat history and risk predictions.
* **Custom UI themes** for a polished experience.

⚠ **Disclaimer:** This is a research/demo tool, not a medical device. Consult professionals for actual diagnoses.

---

## 📊 Dataset

* **File:** `diabetes.csv` (Pima Indians Diabetes Dataset or similar)
* **Features:** `Age`, `BMI`, `Glucose`, `BloodPressure`, `Outcome`
* **Label:** `Outcome` — 1 = High Risk, 0 = Low Risk

---

## 🧠 ML & AI Stack

* **ML Model:** `RandomForestClassifier` (scikit-learn)
* **LLMs:**
  * Primary: Google Gemini (`google-generativeai`)
  * Fallback: Hugging Face GPT-2 (`transformers`)
* **PDF Parsing:** PyMuPDF (`fitz`)
* **Frontend:** Streamlit

---

## ✨ Features

* Chat with AI assistant (PDF report optional)
* Predict diabetes risk from user input
* PDF medical report upload & extraction
* Download chat + prediction report as `.txt`
* Model accuracy metrics view
* Theme toggle (Green & White / Red & White)

---

## ⚙️ Installation

```bash
git clone https://github.com/AkarshYash/HealthGuard-AI
cd healthguard-ai
python -m venv venv
# Activate env
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
````

**Create `.env` file:**

```env
API_KEY=your_google_gemini_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

Place `diabetes.csv` in the root directory.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`.

---

## 📂 requirements.txt

```
streamlit
python-dotenv
google-generativeai
pandas
scikit-learn
transformers
huggingface_hub
torch
PyMuPDF
```

---

## 🖥 Usage

1. **Upload PDF**: Sidebar → Upload your medical report.
2. **Chat**: Ask any health-related question.
3. **Predict Risk**: Enter Age, BMI, Glucose, Blood Pressure → Click Predict.
4. **Download Report**: Save chat + prediction results.

---

## 🛡 Privacy

* PDF text processed locally unless Gemini API is used.
* Do not share personal health data without consent.

---

## 📈 Roadmap

* Add SHAP explainability for predictions.
* Enhance EHR parsing.
* User accounts + encrypted data storage.
* HTML/PDF styled reports.

---

## 📸 Screenshots

Home
![WhatsApp Image 2025-08-10 at 09 48 59\_805e18c0](https://github.com/user-attachments/assets/dec96893-2514-40a4-8a59-20fbc29cae63)

Chat
![WhatsApp Image 2025-08-10 at 09 54 46\_6416a6f0](https://github.com/user-attachments/assets/3badc3d8-864c-4b95-82d7-a52bd0683729)

Prediction
![WhatsApp Image 2025-08-10 at 09 52 11\_deba6bb8](https://github.com/user-attachments/assets/13355e82-6432-4a69-ace4-55294a88f5cd)

[https://github.com/user-attachments/assets/2534c782-339d-45d1-b8d1-421e55fb37d7](https://github.com/user-attachments/assets/2534c782-339d-45d1-b8d1-421e55fb37d7)

---

## 📜 License

MIT License

**Akarsh Chaturvedi** — Cyber Security & Developer
🔗 [LinkedIn Profile](https://www.linkedin.com/in/akarsh-chaturvedi-259271236?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BBrGww08AQLqtNgnrNvLPDg%3D%3D)


