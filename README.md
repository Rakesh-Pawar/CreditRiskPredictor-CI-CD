# CreditRiskPredictor-CI-CD

### Software
0. [Anaconda](https://www.anaconda.com/download)
1. [Github Accoount](https://github.com)
2. [Pycham IDE](https://www.jetbrains.com/pycharm/)
3. [GitCLI](https://git-scm.com/docs/gitcli)
4. [Docker Hub](https://hub.docker.com/)

Create a new environment.
```
conda create -p venv python==3.10 -y
```
Run ML Application on Docker Desktop:
- Deploy the Application on Docker Hub, create the docker container image, and pull in the desktop.
Run Application on Docker Hub

Note: Pull docker image
https://hub.docker.com/r/rakeshpawar20/my-app01/tags

# ML Credit Risk Application

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Phases](#project-phases)
    1. [Problem Definition](#problem-definition)
    2. [Data Collection](#data-collection)
    3. [Data Exploration and Analysis (EDA)](#data-exploration-and-analysis-eda)
    4. [Data Preprocessing](#data-preprocessing)
    5. [Model Development](#model-development)
    6. [Training the Model](#training-the-model)
    7. [Model Evaluation](#model-evaluation)
    8. [Deployment](#deployment)
    9. [Continuous Monitoring and Improvement](#continuous-monitoring-and-improvement)
3. [Results and Impact](#results-and-impact)
4. [Key Learnings](#key-learnings)
5. [ML Model](#ml-model)
    1. [Data Preprocessing](#data-preprocessing-1)
    2. [Feature Engineering](#feature-engineering)
    3. [Model Development](#model-development-1)
        - [Logistic Regression](#logistic-regression)
        - [Random Forest](#random-forest)
6. [Learnings](#learnings)
7. [PyCharm Application](#pycharm-application)
    1. [Imported Necessary Libraries](#imported-necessary-libraries)
    2. [Loaded the Pre-Trained Model](#loaded-the-pre-trained-model)
    3. [Created a Flask Web Application](#created-a-flask-web-application)
    4. [Defined Routes](#defined-routes)
    5. [Home Page (home.html)](#home-page-homehtml)
    6. [Prediction Page (predict route)](#prediction-page-predict-route)
    7. [Result Page (result.html)](#result-page-resulthtml)
    8. [Application Run](#application-run)
---

## Project Overview

### Tools Used:
- Python 🐍
- Machine Learning Libraries 🤖
- PyCharm (Integrated Development Environment) 🌟

### Workflow Diagram:

![image](https://github.com/Rakesh-Pawar/End-to-End-Credit-Risk-Prediction/assets/112051343/5bce6112-3031-4b3b-81fd-54d12b827014)


## Project Phases

### Problem Definition:
Clearly defined the problem: Predict the likelihood of default on credit payments.
Established the importance of accurate credit risk prediction in financial decision-making.

### Data Collection:
Gathered relevant data, including historical credit payment information, applicant details, and other factors influencing credit risk.
Ensured the data's quality and completeness for effective model training.

### Data Exploration and Analysis (EDA):
Conducted Exploratory Data Analysis (EDA) to understand the dataset's structure and characteristics.
Used statistical tests to identify patterns and relationships between variables.
Explored features that significantly contribute to predicting creditworthiness.

### Data Preprocessing:
Handled missing values, outliers, and irrelevant features.
Performed feature scaling and normalization to bring consistency to the data.
Encoded categorical variables and transformed data for model compatibility.

### Model Development:
Selected appropriate machine learning algorithms based on the nature of the problem (classification for credit risk prediction).
Split the dataset into training and testing sets for model training and evaluation.
Employed techniques such as cross-validation to optimize model hyperparameters.

### Training the Model:
Trained the machine learning model using the training dataset.
Fine-tuned the model to improve its accuracy and generalization on unseen data.

### Model Evaluation:
Evaluated the model's performance on the testing dataset using relevant metrics (e.g., accuracy, precision, recall, F1-score).
Conducted a thorough analysis of model strengths and weaknesses.

### Deployment:
Deployed the trained model for real-time credit risk predictions.
Integrated the model into the existing credit decision-making system.

### Continuous Monitoring and Improvement:
Implemented monitoring mechanisms to track model performance over time.
Incorporated feedback and additional data to continually improve the model's predictive accuracy.

## Results and Impact

Successfully built and deployed a machine learning model capable of predicting creditworthiness.
Leveraged data analysis, machine learning, and predictive analytics to enhance lending decisions.
Contributed to reducing credit risk, improving overall profitability, and ensuring more reliable lending practices.

## Key Learnings

Gained expertise in data analysis, machine learning, and predictive analytics through the project.
Applied skills to real-world problem-solving, addressing challenges in the financial domain.
Demonstrated the practical application of machine learning in improving business processes.

## Dataset Overview 

- 0 = Low credit risk i.e high chance of paying back the loan amount
- 1 = High credit risk i.e low chance of paying back the loan amount

### 1. applicant_data:
This file contains personal data about the (primary) applicant

- Unique ID: applicant_id (string)
- Other fields:
  - Primary_applicant_age_in_years (numeric)
  - Gender (string)
  - Marital_status (string)
  - Number_of_dependents (numeric)
  - Housing (string)
  - Years_at_current_residence (numeric)
  - Employment_status (string)
  - Has_been_employed_for_at_least (string)
  - Has_been_employed_for_at_most (string)
  - Telephone (string)
  - Foreign_worker (numeric)
  - Savings_account_balance (string)
  - Balance_in_existing_bank_account_(lower_limit_of_bucket) (string)
  - Balance_in_existing_bank_account_(upper_limit_of_bucket) (string)

### 2. loan_data:
This file contains data more specific to the loan application
- Target: high_risk_application (numeric)
- Other fields:
  - applicant_id (string)
  - Months_loan_taken_for (numeric)
  - Purpose (string)
  - Principal_loan_amount (numeric)
  - EMI_rate_in_percentage_of_disposable_income (numeric)
  - Property (string)
  - Has_coapplicant (numeric)
  - Has_guarantor (numeric)
  - Other_EMI_plans (string)
  - Number_of_existing_loans_at_this_bank (numeric)
  - Loan_history (string)

## ML Model

### Data Preprocessing:
- Null values in the applicant_data were addressed by removing columns with more than 45% null values and imputing remaining null values with the mode.
- Data cleaning involved handling duplicate values, standardizing certain columns, and converting data types.
- In the loan_data, columns with more than 80% null values were dropped, and the remaining null values were imputed using the mode.

### Feature Engineering:
- Unnecessary or duplicate features were dropped.
- Dummies were created for categorical variables.
- The dataset was split into features (X) and the target variable (Y).
- Train-test split was performed.

### Model Development:

#### Logistic Regression:
- Achieved an accuracy score.
- Generated a confusion matrix and a classification report.

#### Random Forest:
- Employed a Random Forest classifier.
- Produced an accuracy score, a confusion matrix, and a classification report.
- Feature importance was analyzed, and the model was exported for future predictions.

## Learnings

- Explored and preprocessed data to ensure quality and completeness.
- Utilized machine learning techniques, including Logistic Regression and Random Forest, to predict credit risk.
- Evaluated model performance using metrics like accuracy, precision, recall, and the confusion matrix.
- Conducted feature engineering to enhance model effectiveness.
- Exported the Random Forest model for potential future use.

## PyCharm Application

### Imported Necessary Libraries:
- pickle: Used for loading the pre-trained Random Forest machine learning model.
- Flask: The web framework used to build the application.
- pandas, numpy, sklearn: Libraries for data manipulation and machine learning.

### Loaded the Pre-Trained Model:
- The pre-trained Random Forest model (rf_model.pkl) was loaded using the pickle module.

### Created a Flask Web Application:
- An instance of the Flask web application was created.

### Defined Routes:
- Two routes were defined:
  - '/' (home): Renders the home page (home.html).
  - '/predict' (predict): Handles form submission, collects user input, and renders the result page (result.html).

### Home Page (home.html):
- A basic HTML template for the home page, which allows users to navigate to the prediction page.

### Prediction Page (predict route):
- A form (In HTML page) is presented to the user, collecting various input features such as marital status, gender, number of dependents, etc.
- Upon form submission, the entered data is collected and converted into a dictionary (new_data).
- The input values are then converted to a NumPy array and reshaped to match the model's input format.
- The model predicts the credit risk based on the input values.
- The prediction result is displayed on the result page (result.html).

### Result Page (result.html):
- The result page shows the outcome of the credit risk prediction, indicating whether the applicant has low or high credit risk.
- The result is dynamically generated based on the model's prediction.

### Application Run:
- The Flask application is run with debugging enabled, and it listens on host 0.0.0.0 at port 50000.

### Additional User Interaction:

1. **User Fills Loan Application Form:**
   - The user accesses the application through a web browser.
   - Navigates to the loan application form.
   - Fills in necessary details such as personal information, loan amount, purpose, etc.
   - Submits the form to the application.

2. **Model Predicts the Applicant Score:**
   - The entered data from the loan application form is processed by the deployed machine learning model.
   - The model predicts the credit risk score based on the provided information.

3. **Bank Decision to Accept Loan Application or Not:**
   - The application incorporates the predicted credit risk score into the decision-making process.
   - The bank's decision algorithm evaluates the credit risk score against predefined thresholds.
   - The system generates a decision to accept or reject the loan application based on the model prediction and business rules.

### Launching the Application:
Used Docker Desktop to run applications.
Open CMD on Project Directory 

- pull docker image if not pull already
docker-compose -f compose.yaml up

- pull public docker image - public Image 
docker pull rakeshpawar20/my-app01:6e4ee26d122ef621fad60fce65effb42a6c2dded

- Run application on docker 
run in cmd: docker-compose -f compose.yaml up 

- Show all images 
docker ps -a

container is running and image name is shown rakeshpawar20/myapp01

You can now open the link given in CMD OR Open from docker desktop 
paste on any browser 
http://localhost:5000/

The application is launched, and users can access it...

---

## 🎉 Acknowledgments and Thanks

Thank you for taking the time to explore my ML Credit Risk Application Project! 



