## Customer Churn Classification Project

### Overview

This project aims to classify customers into **churn** and **no churn** categories using several classification algorithms. It demonstrates a full machine learning pipeline including data preprocessing, model training, evaluation, and deployment.


### Machine Learning Models Used

A range of classification algorithms were implemented and compared:

* **Logistic Regression**
* **Random Forest**
* **XGBoost**
* **Gradient Boosting**

Each model was trained on three variations of the dataset:

* **Oversampled** (to handle class imbalance)
* **Original (no sampling)**
* **Undersampled** (to handle class imbalance)


### Data Transformation Pipeline

To prepare the data for modeling, the following preprocessing steps were applied:

* ✅ **Handling class imbalance** with oversampling and undersampling techniques
* ✅ **Feature scaling** using appropriate scalers
* ✅ **Categorical encoding** with suitable encoding strategies
* ✅ **Saving all preprocessing transformers** to ensure consistent inference-time transformations


### Model Experimentation & Tracking

* All models were trained using multiple **hyperparameter configurations**
* Each experiment was **tracked using [MLflow](https://mlflow.org/)**:

  * Metrics
  * Parameters
  * Model artifacts



### Model Selection & Deployment

* The **best-performing model** was selected and registered in the MLflow Model Registry
* This model is exposed via a **REST API endpoint** served by MLflow



### Streamlit User Interface

An interactive **Streamlit** app was developed as a frontend interface. It allows:

* Making predictions via:
  *  MLflow REST API
  *  Locally saved model
*  Reproducing exact training pipeline via saved preprocessing transformers

---

Here’s your updated section written in **GitHub Markdown format** with proper steps and structure:

---

## Run the Project Locally

To run this project on your local machine, follow the steps below:

###  1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

###  2. Set Up Virtual Environment

Make sure you're using a virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate   # For Windows
# source venv/bin/activate   # For macOS/Linux
```

Install all dependencies:

```bash
pip install -r requirements.txt
```


###  3. Start the MLflow Tracking Server

Open a terminal and run:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000
```

* This will launch the MLflow UI at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### 4. Train and Log Models

In a **new terminal**, run:

```bash
python model.py
```

Note:

* Ensure `model.py` points to MLflow at port `5000`.
* If you're using a different port, **update it inside the code accordingly**.

This will:

* Train multiple models
* Track experiments in MLflow UI
* Automatically register the best model in the **Model Registry**


### 5. Serve the Best Model via MLflow

In another terminal, set the MLflow tracking URI:

```bash
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000      # Windows
# export MLFLOW_TRACKING_URI=http://127.0.0.1:5000   # macOS/Linux
```

Then, serve the registered model:

```bash
mlflow models serve -m "models:/Best_Model_Classifier/1" -p 1234 --no-conda
```

* `1234` is the API port
* `Best_Model_Classifier` is the registered model name
* `1` is the version

This starts the model REST API at: [http://127.0.0.1:1234/invocations](http://127.0.0.1:1234/invocations)


### 6. Launch the Streamlit Application

In a final terminal, run:

```bash
streamlit run app.py
```

This will:

* Launch the Streamlit UI
* Call predictions from the MLflow model endpoint or
* Optionally load a **local model** for inference

---

### Important Notes

* Ensure `.pkl` encoder and transformer files are in the same folder as `app.py`
* Your virtual environment must have all dependencies from `requirements.txt`
* If you encounter port conflicts, feel free to update the ports used (5000 for MLflow, 1234 for model API)

---
