# BeneficiAI – AI-Powered Beneficiary Eligibility System

Developed as part of the software engineering internship program at **InfineCodeTech**.

---

### Core Technologies and Deployment Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly" />
  <img src="https://img.shields.io/badge/Joblib-0052CC?style=for-the-badge&logo=python&logoColor=white" alt="Joblib" />
</p>

---

## Project Overview

BeneficiAI is a data-driven welfare decision-support application built for non-governmental organizations (NGOs) to automate and audit the verification of beneficiary applicants. Evaluating whether a household qualifies for financial aid, educational sponsorships, or medical support is traditionally a manual, slow, and subjective process.

This system replaces manual evaluations with a machine learning classifier integrated directly with a relational database. Field workers can manage beneficiary profiles in real time, monitor demographic trends on an analytics dashboard, train predictive models on fresh database records, and run model inference with probability confidence scores.

## Problem Statement

Welfare distribution programs managed by NGOs frequently suffer from operational inefficiencies:
* **Manual Verification Overhead:** Assessing hundreds of applications manually leads to delayed resource distribution.
* **Assessment Inconsistency:** Different field officers may apply guidelines subjectively, leading to unequal aid distribution.
* **Lack of Audit Trails:** Most workflows fail to log predictive decisions or the criteria used for approvals, preventing compliance and transparency.
* **Inflexible Decision Rules:** Simple static criteria cannot adapt as local socioeconomic parameters and regional demographics shift.

## The Solution

BeneficiAI addresses these issues by introducing an integrated predictive system:
* **Relational Storage:** Maintains a structured beneficiary database to eliminate loose spreadsheets and physical records.
* **AI-Assisted Decisions:** Employs a Random Forest Classifier trained on cleaned applicant metrics to verify eligibility instantly.
* **Audit and Compliance:** Log every prediction, confidence score, and input parameter to an SQL table, creating an unalterable history.
* **Explainable Recommendations:** Pairs machine learning classification with a baseline vulnerability heuristic to explain recommendations to field workers.

---

## Key Features

### Beneficiary Management (CRUD Console)
* **Creation:** Field workers can register new applicants through a validated form. An underlying heuristic automatically calculates a baseline eligibility status.
* **Read and Query:** Provides a searchable table with server-side filters for eligibility status, education levels, employment statuses, and income ranges.
* **Updates and Deletions:** Allows editing of existing records or deleting them through a secondary confirmation safety check to prevent accidental loss.
* **Exporting:** Supports one-click CSV exporting of filtered views for offline reports.

### Demographic & Eligibility Dashboard
* **KPI Metrics:** Renders cards for total applicants, eligible vs. ineligible counts, and average family income.
* **Interactive Visualization:** Integrates Plotly charts including:
  * Eligibility distribution donut charts.
  * Stratified family income histograms.
  * Educational levels grouped by eligibility status.
  * Employment status breakdowns.
  * Age group distribution charts.

### Self-Contained Machine Learning Portal
* **One-Click Training:** Executes the data preparation, scaling, encoding, and training pipeline directly from the user interface.
* **Data Audits:** Displays data cleaning metrics (e.g., number of duplicate rows or missing values removed).
* **Metrics Console:** Reports Accuracy (train vs. test), Precision, Recall, F1-Score, ROC-AUC, and 5-Fold Cross-Validation statistics.
* **Interactive Diagnostics:** Renders a visual confusion matrix heatmap and a feature importance plot to explain which variables (e.g., income, family size) drive predictions.

### Real-Time Prediction and Confidence Gauge
* **Inference Console:** Field workers can enter attributes of a new applicant to get an immediate qualification status.
* **Probability Output:** Renders the class probabilities (`predict_proba`) as a color-coded gauge and confidence percentage.
* **Transparency Layer:** Displays a breakdown of vulnerability flags to cross-reference the machine learning output against local policy benchmarks.

---

## Technology Stack

| Technology | Role in Project | Rationale |
| :--- | :--- | :--- |
| **Python** | Primary Programming Language | Core ecosystem for machine learning, data processing, and scripting. |
| **Streamlit** | Frontend Application Framework | Facilitates fast development of responsive, stateful web applications. |
| **SQLite** | Database Management System | Zero-configuration, serverless relational database for local testing and deployment. |
| **Pandas & NumPy** | Data Manipulation & Vectorization | High-performance manipulation, cleaning, and preprocessing of tabular datasets. |
| **Scikit-Learn** | Machine Learning Engine | Provides implementations of the Random Forest Classifier, model metrics, and data splitting utilities. |
| **Plotly** | Visualization Library | Generates interactive, zoomable data charts with clean hover text. |
| **Joblib** | Model Serialization | Efficiently saves and loads the trained model bundle and LabelEncoder states to disk. |
| **Matplotlib & Seaborn** | Backup Visualization | Used for static data visualization rendering during background model testing. |

---

## Machine Learning Pipeline

```
[Raw SQL Data] ──► [Data Cleaning] ──► [Fixed Label Encoding] ──► [Stratified Split] ──► [Model Fitting] ──► [Serialization]
```

The system implements an automated pipeline composed of the following stages:

1. **Database Ingestion:** Queries the SQLite database dynamically. This ensures that any additions or updates made by field officers are immediately reflected in subsequent model training runs.
2. **Preprocessing and Data Cleaning:**
   * Eliminates exact duplicate records.
   * Cleans categorical string column values and strips leading or trailing whitespace.
   * Drops rows containing missing values (`NaN`) in key features.
   * Clamps outlier income values to valid ranges.
3. **Fixed-Vocabulary Label Encoding:** Rather than fitting encoders on the training split dynamically, the system pre-populates category vocabularies. This ensures that encodings are consistent between training sessions and production inference, preventing errors when a category is missing in the training split.
4. **Stratified Split:** Splits the preprocessed data into an 80% training set and a 20% test set, maintaining identical target class ratios in both splits.
5. **Model Training:** Fits a `RandomForestClassifier` configured with balanced class weights to compensate for demographic skews in applicant records.
6. **Performance Evaluation:** Evaluates test predictions to compile accuracy, precision, recall, F1, and ROC-AUC metrics. Runs a 5-fold cross-validation routine to assess generalization.
7. **Model Persistence:** Packages the trained classifier, fixed encoders, and baseline performance metrics into a joblib bundle saved at `models/eligibility_model.pkl`.

---

## Model Evaluation

The model trained on the baseline dataset of 300 synthetic records achieves the following test performance:

### Performance Metrics
* **Training Accuracy:** 100.0%
* **Test Accuracy:** 95.0%
* **Precision (Eligible Class):** 97.4%
* **Recall (Eligible Class):** 95.0%
* **F1-Score:** 96.2%
* **Area Under ROC Curve (ROC-AUC):** 0.989
* **5-Fold Cross-Validation Accuracy:** 94.2% (±1.5%)

### Confusion Matrix
```
                 Predicted Ineligible | Predicted Eligible
Actual Ineligible         19          |         1
Actual Eligible            2          |        38
```

### Feature Importance Analysis
Using Mean Decrease in Gini Impurity, the Random Forest model identifies the following ranking for eligibility prediction:
1. **Family Income:** 42.1% (Primary economic discriminator)
2. **Family Members:** 21.4% (Directly influences per-capita resource availability)
3. **Age:** 15.3% (Proxy for working capacity)
4. **Employment Status:** 10.8% (Indicates household stability)
5. **Education Level:** 7.2% (Correlates with long-term earning potential)
6. **Disability Status:** 3.2% (Direct vulnerability flag)

---

## Screenshots

*(Placeholders for application interface demonstrations)*

### 1. Real-Time Analytics Dashboard
```
[ Placeholder: Dashboard view displaying KPI cards and dynamic demographic charts ]
```

### 2. CRUD Beneficiary Management
```
[ Placeholder: CRUD table view with active filters, edit modals, and CSV export action ]
```

### 3. Model Training Diagnostics
```
[ Placeholder: Model training interface with metrics, confusion matrix, and feature importances ]
```

### 4. Eligibility Inference Console
```
[ Placeholder: Live prediction page showing input form, confidence gauge, and vulnerability details ]
```

---

## Installation & Setup

### Prerequisites
* Python 3.10 or higher
* SQLite 3

### 1. Clone the Project
```bash
git clone https://github.com/ik4rthik/BenefiAI.git
cd BenefiAI
```

### 2. Configure Virtual Environment
Create a clean environment to isolate package dependencies:
```bash
python -m venv venv

# Activate on Windows (PowerShell)
venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize Database and Generate Dataset
Generate the synthetic dataset containing 300 applicant records and load it into the SQLite database:
```bash
# Generate the CSV data
python data/generate_dataset.py

# Create the SQLite tables and insert initial records
python database/db_setup.py
```

### 5. Run Verification Tests
Verify the installation by running the machine learning pipeline smoke test:
```bash
python test_ml.py
```

### 6. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## Challenges Solved

### 1. Prevention of Training-Serving Categorical Drift
* **The Challenge:** Scikit-Learn's `LabelEncoder` can generate inconsistent maps if a categorical label appears in the test split but is absent in the training split, resulting in runtime errors during inference.
* **The Solution:** The pipeline uses pre-defined categorical vocabularies mapping values directly to indices. The label encoders are pre-configured with these vocabularies to guarantee identical integer codes during training and real-time prediction.

### 2. Imbalanced Data and Classification Bias
* **The Challenge:** Real-world beneficiary databases often display class imbalance. A standard classifier trained on this data might prioritize the majority class, leading to low recall for eligible applicants.
* **The Solution:** We set `class_weight="balanced"` in the Random Forest configuration, adjusting training weights inversely proportional to class frequencies to prevent bias and maintain high recall.

### 3. Streamlit Session and Relational Database Synchronization
* **The Challenge:** Streamlit applications re-run the entire script upon user interaction. Managing database connections without leaks while maintaining visual state across pages can be difficult.
* **The Solution:** We established connection context managers in `database/crud.py` to open and close connections safely. The ML pipeline queries SQLite directly to ensure models are trained on active beneficiary records.

### 4. Explainable AI (XAI) in Welfare Decisions
* **The Challenge:** Predictive models can behave as black boxes, making it difficult for field officers to trust or audit eligibility recommendations.
* **The Solution:** The prediction view presents a hybrid output. It pairs the model's probabilistic confidence score with a deterministic vulnerability flag checklist, explaining exactly why an applicant is flagged as eligible.

---

## Future Improvements

* **Role-Based Access Control (RBAC):** Introduce a login system with distinct read/write permissions for field workers and administrative roles.
* **Automated Data Validation APIs:** Connect the interface to official public registries (e.g., identity or tax systems) to verify applicant details automatically.
* **Inference Drift Monitoring:** Create an dashboard to monitor prediction distributions over time and alert admins if covariate shift is detected.
* **Containerization:** Package the application using Docker to facilitate deployment to AWS, GCP, or Azure.

---

## Internship Learning Outcomes

During this project at **InfineCodeTech**, I gained practical software engineering and data science competencies:
* **Production ML Systems:** Learned to build modular prediction systems separating dataset preparation, database configuration, model training, and prediction modules.
* **Training-Serving Skew Mitigation:** Understood and resolved common engineering issues like category vocabulary misalignment between offline training and online prediction.
* **Relational Database Integration:** Designed SQL schemas, handled CRUD queries, and built prediction logging tables in SQLite to support real-time state tracking.
* **State Management & UI Design:** Developed multi-page, stateful Streamlit applications integrated with Plotly Express to visualize live database state.
* **Testing and Clean Code:** Practiced test-driven concepts by writing pipeline validation scripts and organizing project files to make the codebase maintainable.
