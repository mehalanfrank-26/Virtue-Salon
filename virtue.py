import pandas as pd
from collections import Counter
from datetime import datetime, timedelta

# Load sample data
df = pd.read_csv("salon_customer_data.csv", parse_dates=["date"])

# Define cutoff date for "current" time
current_date = pd.to_datetime("2025-07-01")

# Function to analyze each customer's service history
def analyze_customer_behavior(customer_df):
    customer_df = customer_df.sort_values("date")
    
    services = customer_df["service"].tolist()
    service_counts = Counter(services)
    
    frequent_services = [s for s, c in service_counts.items() if c >= 3]
    rare_services = [s for s, c in service_counts.items() if c == 1]
    
    all_categories = ["Hair", "Skin", "Nails", "Hair Removal", "Wellness"]
    tried_categories = customer_df["Category"].unique().tolist()
    inactive_categories = [c for c in all_categories if c not in tried_categories]
    
    last_visit = customer_df["date"].max()
    expected_next = last_visit + timedelta(days=30)
    
    missed_visit = current_date > expected_next
    
    return {
        "frequent_services": frequent_services,
        "rare_services": rare_services,
        "inactive_categories": inactive_categories,
        "last_visit": last_visit.strftime("%Y-%m-%d"),
        "missed_visit": missed_visit
    }

# Apply for each customer
customers = df["customer_id"].unique()
results = []

for cid in customers:
    cust_df = df[df["customer_id"] == cid]
    behavior = analyze_customer_behavior(cust_df)
    
    result = {
        "customer_id": cid,
        "name": cust_df["name"].iloc[0],
        "email": cust_df["email"].iloc[0],
        "phone": cust_df["phone"].iloc[0],
        **behavior
    }
    results.append(result)

result_df = pd.DataFrame(results)

# Show result
print(result_df)

# Save for LLM prompt generation
result_df.to_json("llm_prompts_input.json", orient="records", indent=2)
