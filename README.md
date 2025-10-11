# MLA_Firewall_Anomaly
Anomaly Detection And Attack Prediction From Firewall Logs


# TO TRAIN THE MODEL
python ./train_model.py --train_csv ./firewall_train.csv --outdir ./


# TO EVALUATE THE MODEL
python ./test_model.py --test_csv ./firewall_test.csv --iso_model ./model_isolation_forest.joblib --clf_model ./model_gb_classifier.joblib --outdir ./ 


# TO PREDICT USING REAL TIME DATA
python ./predict.py --input_csv ./firewall_test.csv --iso_model ./model_isolation_forest.joblib --clf_model ./model_gb_classifier.joblib --out ./predictions.csv
