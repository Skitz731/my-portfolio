import random
import csv
import os
import sys

def generate_churn_data(n_rows, output_path, seed=42):
    """Generate a synthetic customer churn dataset."""
    random.seed(seed)

    header = [
        "customer_id", "age", "gender", "tenure_months",
        "monthly_charges", "total_charges", "contract_type",
        "payment_method", "num_support_tickets", "churned"
    ]

    contract_types = ["month-to-month", "one-year", "two-year"]
    payment_methods = ["credit_card", "bank_transfer", "electronic_check", "mailed_check"]

    rows = []
    for i in range(n_rows):
        tenure = random.randint(1, 72)
        monthly = round(random.uniform(20, 120), 2)
        contract = random.choice(contract_types)

        # Make churn more likely for month-to-month, high charges, low tenure
        churn_prob = 0.15
        if contract == "month-to-month":
            churn_prob += 0.15
        if monthly > 80:
            churn_prob += 0.10
        if tenure < 12:
            churn_prob += 0.10

        churned = 1 if random.random() < churn_prob else 0

        row = [
            f"CUST-{i+1:05d}",
            random.randint(18, 75),
            random.choice(["M", "F"]),
            tenure,
            monthly,
            round(monthly * tenure + random.uniform(-50, 50), 2),
            contract,
            random.choice(payment_methods),
            random.randint(0, 10),
            churned
        ]
        rows.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    churn_count = sum(r[-1] for r in rows)
    print(f"Generated {n_rows} rows at {output_path}")
    print(f"Churn rate: {churn_count/n_rows:.1%} ({churn_count} churned)")

if __name__ == "__main__":
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/customers.csv"
    generate_churn_data(n_rows, output_path)