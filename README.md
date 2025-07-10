# 🎯 Career Path Predictor

An intelligent machine learning system that recommends suitable career paths for computer science students based on their academic data, interests, and demographics. Developed as part of the **AICTE-Edunet-Microsoft AI Internship**.

![Confusion Matrix](assets/confusion_matrix.png) <!-- Add your actual image if available -->

---

## 🧠 Problem Statement

Many students face difficulty in choosing the right career path due to the lack of personalized guidance. This project addresses this challenge by building a predictive model that helps students identify ideal career options based on key attributes like GPA, age, gender, and domain interests.

---

## ✅ Objectives

- Predict future career paths using machine learning
- Handle imbalanced class distributions and categorical data
- Visualize model performance using classification reports and confusion matrices
- Build a scalable, interpretable system that can be extended into real-world applications

---

## 📂 Dataset

- **Title**: [Computer Science Students Career Prediction](https://www.kaggle.com/datasets/siddharthm1698/computer-science-students-career-prediction)  
- **Source**: Kaggle  
- **Features**:
  - `Age`
  - `GPA`
  - `Gender`
  - `Interested Domain`
  - `Future Career` (target variable)

---

## 🛠️ Technologies Used

- Python
- scikit-learn
- pandas
- matplotlib
- seaborn
- Google Colab / Jupyter Notebook

---

## ⚙️ Model Architecture

- **Preprocessing**:
  - `StandardScaler` for numerical features (GPA, Age)
  - `OneHotEncoder` for categorical features (Gender, Interested Domain)
  - Rare classes in the target variable grouped as "Other"
- **Model**: `RandomForestClassifier`
  - `max_depth=5`
  - `class_weight='balanced'` to handle class imbalance
- **Evaluation**: 
  - `classification_report`
  - `confusion_matrix` with heatmap visualization

---

## 📈 Results

- ✅ **Test Accuracy**: **92%**
- 🔍 High precision and recall across most career classes
- 📊 Strong generalization to unseen data

---

## 🔮 Future Improvements

- Integrate more features: skill levels, certifications, extracurriculars, personality traits
- Build a user-facing interface with **Streamlit** or **Flask**
- Expand dataset for broader generalization across disciplines
- Use ensemble models or deep learning for improved performance

---

## 📁 Folder Structure

📦career-path-predictor
┣ 📜career_predictor.py
┣ 📂assets
┃ ┗ 📸 confusion_matrix.png
┣ 📜README.md
┗ 📜requirements.txt

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1. Clone the Repository
git clone https://github.com/autumnByte/career-path-predictor.git
cd career-path-predictor
### 2. Install Requirements
Copy
Edit
pip install -r requirements.txt
### 3. Run the Notebook / Script
Open career_predictor.py in Google Colab or Jupyter Notebook.

---

## 📬 Contact
---
-Suhani Dwivedi
- 🎓 B.Tech CSE | KIIT University
-📧 suhanidwivedi2k5@gmail.com
-🌐 [linkedin.com](https://www.linkedin.com/in/suhani-dwivedi)
