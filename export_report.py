# export_report.py
import pandas as pd
from data.storage import get_conn
from evaluation.risk import compute_var

conn = get_conn()

metrics     = pd.read_sql("SELECT * FROM model_metrics ORDER BY trained_at DESC", conn)
predictions = pd.read_sql("SELECT * FROM predictions ORDER BY date DESC", conn)
prices      = pd.read_sql("SELECT * FROM prices", conn)

conn.close()

var_result = compute_var()
var_df = pd.DataFrame([var_result])

with pd.ExcelWriter("equity_research_report.xlsx", engine="openpyxl") as writer:
    metrics.to_excel(writer, sheet_name="Model Comparison", index=False)
    predictions.to_excel(writer, sheet_name="All Predictions", index=False)
    var_df.to_excel(writer, sheet_name="Risk (VaR)", index=False)

    # Latest BUY/SELL signals summary
    latest = predictions.sort_values("date").groupby("ticker").last().reset_index()
    latest[["ticker","predicted_return","actual_return","signal"]]\
        .sort_values("predicted_return", ascending=False)\
        .to_excel(writer, sheet_name="Latest Signals", index=False)

print("Report saved: equity_research_report.xlsx")