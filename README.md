> **"Smarter Health Starts With Smarter AI Predictions 🧠❤️"**

# Heart Disease Prediction Model 🩺🔍

A complete classification-based machine learning pipeline designed to predict the likelihood of heart disease using real-world clinical data. This project features robust preprocessing, model tuning, evaluation, and deployment using modern tools like Flask and Streamlit.

---

## 📁 Repository Structure

```
.
├── Data/                   # Raw and processed datasets
│   └── processed_hungarian.csv
├── Output/                 # Evaluation results
│   └── confusion_matrix.png
├── Models/                 # Saved model and scaler
│   ├── heart_disease_model.pkl
│   └── scaler.pkl
├── Notebook/               # Jupyter notebooks for development
│   └── model_training.ipynb
├── deployment task/        # Deployment scripts
│   ├── app.py              # Flask API
│   └── streamlit_app.py    # Streamlit frontend
├── requirements.txt        # Python dependencies
└── README.md               # Project overview (this file)
```

---

## 🧠 Key Features
- End-to-end ML pipeline (cleaning → training → saving)
- Multiple models trained and tuned (XGBoost, RF, SVM, etc.)
- Hyperparameter optimization with Grid and Random Search
- Evaluation with accuracy, precision, recall, F1-score
- Confusion matrix visualization saved to `/Output`
- Flask API + Streamlit app for interactive predictions

---

## 📊 Model Performance
| Metric     | Score     |
|------------|-----------|
| Accuracy   | 0.91      |
| Precision  | 0.92 (Class 1) |
| Recall     | 0.83 (Class 1) |
| F1-Score   | 0.87 (Class 1) |

✅ Evaluation saved as: `Output/confusion_matrix.png`
📄 Report: `docs/random_search_report.txt`

---

## 🚀 How to Run the Project

### 🔧 Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### 🧪 Step 2: Run the Flask API (Local Deployment)
```bash
python deployment task/app.py
```

### 🖥️ Step 3: Run the Streamlit App (Frontend)
```bash
streamlit run deployment task/streamlit_app.py
```

---

## 📦 Model Usage in Code
```python
import pickle, pandas as pd
model = pickle.load(open("Models/heart_disease_model.pkl", "rb"))
scaler = pickle.load(open("Models/scaler.pkl", "rb"))
input_data = pd.DataFrame([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]], columns=[...])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
```

---

## 📈 Future Improvements
- Host on Streamlit Cloud / Render
- Add patient-level visualization insights
- Extend to multi-class risk levels
- Deploy with Docker or CI/CD pipelines

---

## 👤 Author
- **Peter Olamojin**
- Email: olapeter1010@gmail.com
