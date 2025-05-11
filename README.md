# Insurance Claim Fraud Detection

Benchmark of four classical ML models on an imbalanced auto‑insurance‑claim
data set.  Best model (XGBoost + SMOTE) achieves **F1 = 0.65** and **AUC = 0.86**.

 
#Reproduce locally

bash
git clone https://github.com/your‑username/insurance-fraud-detection.git
cd insurance-fraud-detection
pip install -r requirements.txt
python fraud_experiment.py --data path/to/insurance_claims.csv
