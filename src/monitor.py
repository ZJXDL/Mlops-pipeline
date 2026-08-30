import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_drift_check():
    # Load reference (training) data and current inference sample
    reference_data = pd.read_csv("data/processed/train.csv").drop("failure", axis=1)
    current_data = pd.read_csv("data/processed/test.csv").drop("failure", axis=1)

    # Initialize and run Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/data_drift_report.html"
    drift_report.save_html(report_path)
    print(f"Data drift report generated successfully at: {report_path}")

if __name__ == "__main__":
    run_drift_check()