# 🏠 Home Loan Approval Prediction System
<br>
<br>
<br>





A full-stack machine learning web application to predict **Home Loan Approval** using customer financial and demographic data. Built with **Flask**, **scikit-learn**, and a clean, responsive **HTML/CSS** interface.
<br>
<br>




---

## 🚀 Features<br><br>

- ✅ Predict home loan approval with high accuracy<br>
- 📈 Trained using multiple ML models (Logistic Regression, SVM, Random Forest, etc.)<br>
- 🏆 Automatically selects and saves the best-performing model<br>
- 🧼 Cleaned dataset with outlier removal and scaling<br>
- 🌐 User-friendly front-end with stylish design<br>
- 💾 Model saved using `pickle`/`joblib` for deployment<br>

---<br>
<br>
<br>
// 📂 Project Structure<br><br><br>





---


<br><br>


// 🧠 ML Model Training (Jupyter Notebook)<br><br>

The model is trained in `model_training.ipynb` with the following pipeline:<br><br>


<br>
- Load and preprocess CSV data<br>
- Encode categorical features<br>
- Handle missing values and outliers<br>
- Feature scaling using `StandardScaler`<br>
- Train and evaluate:<br>
  - Logistic Regression<br>
  - Decision Tree<br>
  - Random Forest<br>
  - SVM<br>
  - K-Nearest Neighbors (KNN)<br>
- Automatically pick the best model based on accuracy<br>
- Save the model as `best_home_loan_model.pkl`<br>

---<br>





<br><br><br>

//💻 Flask Web App<br>

### 🏠 `index.html`<br>
- Responsive form to enter loan applicant details<br>
- Dropdown for employment type<br>
- Displays prediction results dynamically<br>

### 🔧 `app.py`<br>
- Loads `best_home_loan_model.pkl` using `joblib`<br>
- Accepts form input, performs prediction, returns result to UI<br><br>

---

## 📦 Installation<br><br>





<br>
<br>
<br>


###### Demo Video


<br>
<br>
<br>




https://github.com/user-attachments/assets/9250f263-040d-4bed-893b-52be1bbcc9d3









 
 
