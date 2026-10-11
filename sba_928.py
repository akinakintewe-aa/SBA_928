import json
import zipfile
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "online+retail.zip"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "google/flan-t5-small"

EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)



print("\nREQUIREMENT 1: DATASET PREPARATION")

# Load the Excel dataset from the ZIP archive.

with zipfile.ZipFile(DATA_PATH, "r") as z:

    excel_name = next(
        name for name in z.namelist()
        if name.lower().endswith(".xlsx")
    )

    with z.open(excel_name) as f:
        df = pd.read_excel(BytesIO(f.read()))

print("Original dataset:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Inspect invalid or special transaction records.

negative_quantity = (df["Quantity"] < 0).sum()
negative_price = (df["UnitPrice"] < 0).sum()
zero_price = (df["UnitPrice"] == 0).sum()

cancelled = (
    df["InvoiceNo"]
    .astype(str)
    .str.startswith("C")
    .sum()
)

print("\nNegative quantities:", negative_quantity)
print("Negative prices:", negative_price)
print("Zero prices:", zero_price)
print("Cancelled records:", cancelled)

# Keep positive-value sales for this analysis.
# Preserve the original dataset for bias assessment.
# Missing CustomerID values are retained for sales analysis.

sales = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0) &
    (df["Description"].notna()) &
    (~df["InvoiceNo"].astype(str).str.startswith("C"))
].copy()

sales["Description"] = (
    sales["Description"]
    .astype(str)
    .str.strip()
)

sales["Revenue"] = (
    sales["Quantity"] * sales["UnitPrice"]
)

sales["Month"] = (
    sales["InvoiceDate"]
    .dt.to_period("M")
    .astype(str)
)

print("\nCleaned dataset:", sales.shape)

sales.to_csv(
    RESULTS_DIR / "cleaned_retail.csv",
    index=False
)



# Country-level statistics.

country_stats = sales.groupby("Country").agg(
    Revenue=("Revenue", "sum"),
    Units=("Quantity", "sum"),
    Invoices=("InvoiceNo", "nunique")
)

country_stats["AverageItemsPerInvoice"] = (
    country_stats["Units"] / country_stats["Invoices"]
)

country_stats = country_stats.sort_values(
    "Revenue",
    ascending=False
)

country_stats.to_csv(
    RESULTS_DIR / "country_statistics.csv"
)

# Product-level statistics.

product_stats = sales.groupby("Description").agg(
    Revenue=("Revenue", "sum"),
    Units=("Quantity", "sum")
)

product_stats = product_stats.sort_values(
    "Revenue",
    ascending=False
)

product_stats.to_csv(
    RESULTS_DIR / "product_statistics.csv"
)

# Monthly market trends.

monthly_stats = (
    sales.groupby("Month")["Revenue"]
    .sum()
    .sort_index()
)

monthly_stats.to_csv(
    RESULTS_DIR / "monthly_statistics.csv"
)

print("\nTotal revenue:", round(sales["Revenue"].sum(), 2))

print("\nTop five countries:")
print(country_stats.head())

print("\nTop five products:")
print(product_stats.head())




# Each example contains:
# 1. A research scenario
# 2. Relevant facts from the dataset
# 3. A market research question
# 4. A verified target answer
#
# The facts are provided in the prompt so that both
# the original and fine-tuned models have access to
# the information required to answer.

records = []


def add_record(
    scenario,
    group_type,
    group_id,
    context,
    question,
    answer
):

    records.append({
        "scenario": scenario,
        "group_type": group_type,
        "group_id": group_id,
        "context": context,
        "question": question,
        "answer": answer
    })


def money(value):
    return f"GBP {value:,.2f}"




# Use up to 20 countries for a manageable experiment.
# Each included country contributes the same number
# of country-level research examples.

selected_countries = country_stats.head(20)

for country, row in selected_countries.iterrows():

    context = (
        f"Country: {country}\n"
        f"Total revenue: {money(row['Revenue'])}\n"
        f"Total units: {int(row['Units'])}\n"
        f"Total invoices: {int(row['Invoices'])}\n"
        f"Average items per invoice: "
        f"{row['AverageItemsPerInvoice']:.2f}"
    )

    add_record(
        scenario="Geographic market analysis",
        group_type="country",
        group_id=country,
        context=context,
        question=(
            f"What was the total sales revenue "
            f"in {country}?"
        ),
        answer=money(row["Revenue"])
    )




# Average items per invoice measures purchasing volume.
# It does not establish individual consumer preferences.

for country, row in selected_countries.iterrows():

    context = (
        f"Country: {country}\n"
        f"Total units: {int(row['Units'])}\n"
        f"Total invoices: {int(row['Invoices'])}\n"
        f"Average items per invoice: "
        f"{row['AverageItemsPerInvoice']:.2f}"
    )

    add_record(
        scenario="Consumer purchasing behavior",
        group_type="country",
        group_id=country,
        context=context,
        question=(
            f"What was the average number of "
            f"items per invoice in {country}?"
        ),
        answer=f"{row['AverageItemsPerInvoice']:.2f}"
    )




selected_products = product_stats.head(30)

for product, row in selected_products.iterrows():

    context = (
        f"Product: {product}\n"
        f"Units sold: {int(row['Units'])}\n"
        f"Revenue: {money(row['Revenue'])}"
    )

    add_record(
        scenario="Product demand",
        group_type="product",
        group_id=product,
        context=context,
        question=(
            f"How many units of {product} "
            f"were sold?"
        ),
        answer=str(int(row["Units"]))
    )




months = list(monthly_stats.index)

for i in range(1, len(months)):

    previous_month = months[i - 1]
    current_month = months[i]

    previous_revenue = monthly_stats[previous_month]
    current_revenue = monthly_stats[current_month]

    if current_revenue > previous_revenue:
        trend = "increased"

    elif current_revenue < previous_revenue:
        trend = "decreased"

    else:
        trend = "remained unchanged"

    context = (
        f"Previous month: {previous_month}\n"
        f"Previous revenue: {money(previous_revenue)}\n"
        f"Current month: {current_month}\n"
        f"Current revenue: {money(current_revenue)}"
    )

    add_record(
        scenario="Market trends",
        group_type="month",
        group_id=current_month,
        context=context,
        question=(
            f"Did sales revenue increase, decrease, "
            f"or remain unchanged in {current_month} "
            f"compared with {previous_month}?"
        ),
        answer=trend
    )




# This compares products within the same retailer.
# It does not measure competition between companies.

product_names = list(selected_products.index)

for i in range(0, 20, 2):

    product_a = product_names[i]
    product_b = product_names[i + 1]

    revenue_a = product_stats.loc[product_a, "Revenue"]
    revenue_b = product_stats.loc[product_b, "Revenue"]

    if revenue_a > revenue_b:
        answer = product_a

    elif revenue_b > revenue_a:
        answer = product_b

    else:
        answer = "Equal revenue"

    context = (
        f"Product A: {product_a}\n"
        f"Product A revenue: {money(revenue_a)}\n"
        f"Product B: {product_b}\n"
        f"Product B revenue: {money(revenue_b)}"
    )

    add_record(
        scenario="Internal product competition",
        group_type="comparison",
        group_id=f"pair_{i}",
        context=context,
        question=(
            "Which product generated more revenue?"
        ),
        answer=answer
    )



# The dataset has no external competitor sales.
# An accurate model must not invent market shares.

for country in list(selected_countries.index):

    context = (
        f"Country: {country}\n"
        "Available information: sales transactions "
        "from one retailer.\n"
        "External competitor sales: unavailable.\n"
        "Total market sales: unavailable."
    )

    add_record(
        scenario="External competitor analysis",
        group_type="country",
        group_id=country,
        context=context,
        question=(
            f"What is the retailer's market share "
            f"relative to competitors in {country}?"
        ),
        answer=(
            "Cannot determine market share "
            "from the available data."
        )
    )


print("\nCurated research examples:", len(records))



print("\nREQUIREMENT 2: PROMPT VARIATIONS")


def create_prompt(record, style):

    context = record["context"]
    question = record["question"]

    # Variation 1: Direct prompting.

    if style == "direct":

        return (
            f"Data:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    # Variation 2: Role-based prompting.

    if style == "role":

        return (
            "You are a market research analyst.\n"
            "Use only the supplied research data.\n"
            "Do not invent missing information.\n\n"
            f"Data:\n{context}\n\n"
            f"Question: {question}\n"
            "Give a concise answer:"
        )

    # Variation 3: Structured prompting.

    if style == "structured":

        return (
            "Market Research Task\n\n"
            f"{context}\n\n"
            f"Research question: {question}\n\n"
            "Instructions:\n"
            "1. Use only the supplied data.\n"
            "2. Do not make unsupported assumptions.\n"
            "3. Return only the answer.\n\n"
            "Result:"
        )

    # Fourth wording reserved for evaluation.

    if style == "evaluation":

        return (
            "Analyze the following retail information.\n"
            "Answer using only the provided facts.\n"
            "If the information is insufficient, say so.\n\n"
            f"{context}\n\n"
            f"{question}\n\n"
            "Answer:"
        )

    raise ValueError("Unknown prompt style")




# Group related questions so that the same country,
# product or month is not used in both sets within
# its corresponding group type.

rng = np.random.default_rng(SEED)

train_groups = set()
test_groups = set()

group_types = sorted(
    set(record["group_type"] for record in records)
)

for group_type in group_types:

    groups = sorted({
        record["group_id"]
        for record in records
        if record["group_type"] == group_type
    })

    rng.shuffle(groups)

    split = max(1, int(len(groups) * 0.8))

    if len(groups) > 1:
        split = min(split, len(groups) - 1)

    for group_id in groups[:split]:
        train_groups.add((group_type, group_id))

    for group_id in groups[split:]:
        test_groups.add((group_type, group_id))


train_records = [
    record for record in records
    if (
        record["group_type"],
        record["group_id"]
    ) in train_groups
]

test_records = [
    record for record in records
    if (
        record["group_type"],
        record["group_id"]
    ) in test_groups
]

# Generate three prompt variations for training.

training_examples = []

for record in train_records:

    for style in ["direct", "role", "structured"]:

        training_examples.append({
            "scenario": record["scenario"],
            "style": style,
            "prompt": create_prompt(record, style),
            "answer": record["answer"]
        })


# Keep evaluation examples separate from training.

evaluation_examples = []

for record in test_records:

    evaluation_examples.append({
        "scenario": record["scenario"],
        "style": "evaluation",
        "prompt": create_prompt(
            record,
            "evaluation"
        ),
        "answer": record["answer"]
    })

# Balance the market-trend training examples.

trend_decreases = [
    example for example in training_examples
    if example["scenario"] == "Market trends"
    and example["answer"] == "decreased"
]

training_examples.extend(trend_decreases)

np.random.default_rng(SEED).shuffle(training_examples)

print("Training examples:", len(training_examples))
print("Evaluation examples:", len(evaluation_examples))

# Save the curated prompts.

pd.DataFrame(training_examples).to_csv(
    RESULTS_DIR / "training_prompts.csv",
    index=False
)

pd.DataFrame(evaluation_examples).to_csv(
    RESULTS_DIR / "evaluation_prompts.csv",
    index=False
)




print("\nREQUIREMENT 3: MODEL FINE-TUNING")

# Download the original pretrained model.

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.to(device)



class RetailDataset(torch.utils.data.Dataset):

    def __init__(self, examples):

        self.examples = examples

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        inputs = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=256
        )

        targets = tokenizer(
            text_target=example["answer"],
            truncation=True,
            max_length=64
        )

        inputs["labels"] = targets["input_ids"]

        return inputs


train_dataset = RetailDataset(training_examples)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator
)


def generate_answer(model, prompt):

    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=2,
            do_sample=False
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()



print("\nREQUIREMENT 4: ORIGINAL MODEL EVALUATION")

original_predictions = []

for example in evaluation_examples:

    prediction = generate_answer(
        model,
        example["prompt"]
    )

    original_predictions.append(prediction)

print("Original model evaluation completed.")



optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

training_losses = []

print("\nStarting fine-tuning...")

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for batch in train_loader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)

    training_losses.append({
        "epoch": epoch + 1,
        "average_loss": average_loss
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


# Save the model and training results.

model_directory = RESULTS_DIR / "fine_tuned_model"

model.save_pretrained(model_directory)

tokenizer.save_pretrained(model_directory)

pd.DataFrame(training_losses).to_csv(
    RESULTS_DIR / "training_loss.csv",
    index=False
)

print("\nFine-tuning completed.")




print("\nEvaluating fine-tuned model...")

fine_tuned_predictions = []

for example in evaluation_examples:

    prediction = generate_answer(
        model,
        example["prompt"]
    )

    fine_tuned_predictions.append(prediction)




def normalize_answer(answer):

    return " ".join(
        str(answer)
        .lower()
        .replace(",", "")
        .replace("£", "gbp")
        .strip()
        .split()
    )


comparison = []

for i, example in enumerate(evaluation_examples):

    expected = example["answer"]

    original = original_predictions[i]
    fine_tuned = fine_tuned_predictions[i]

    original_correct = (
        normalize_answer(original)
        == normalize_answer(expected)
    )

    fine_tuned_correct = (
        normalize_answer(fine_tuned)
        == normalize_answer(expected)
    )

    comparison.append({
        "scenario": example["scenario"],
        "prompt": example["prompt"],
        "expected": expected,
        "original_output": original,
        "fine_tuned_output": fine_tuned,
        "original_correct": original_correct,
        "fine_tuned_correct": fine_tuned_correct
    })


comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)


# Overall exact-match accuracy.

original_accuracy = (
    comparison_df["original_correct"].mean() * 100
)

fine_tuned_accuracy = (
    comparison_df["fine_tuned_correct"].mean() * 100
)


print("\nMODEL COMPARISON")

print(
    f"Original accuracy: {original_accuracy:.2f}%"
)

print(
    f"Fine-tuned accuracy: {fine_tuned_accuracy:.2f}%"
)


# Compare accuracy by market research scenario.

scenario_accuracy = (
    comparison_df.groupby("scenario")[
        ["original_correct", "fine_tuned_correct"]
    ]
    .mean() * 100
)

print("\nAccuracy by research scenario:")
print(scenario_accuracy)

scenario_accuracy.to_csv(
    RESULTS_DIR / "scenario_accuracy.csv"
)



print("\nREQUIREMENT 5: BIAS ASSESSMENT")




country_counts = (
    sales["Country"]
    .value_counts()
)

country_share = (
    country_counts / len(sales) * 100
)

country_share.to_csv(
    RESULTS_DIR / "country_representation.csv",
    header=["Percentage"]
)

uk_share = country_share.get(
    "United Kingdom",
    0
)

print(
    f"\nUK transaction representation: "
    f"{uk_share:.2f}%"
)


#

missing_customer = (
    df["CustomerID"]
    .isna()
    .mean() * 100
)

print(
    f"Missing customer identifiers: "
    f"{missing_customer:.2f}%"
)




customer_missing_by_country = (
    sales.assign(
        MissingCustomer=sales["CustomerID"].isna()
    )
    .groupby("Country")["MissingCustomer"]
    .agg(["mean", "size"])
)

customer_missing_by_country["mean"] *= 100

customer_missing_by_country.columns = [
    "MissingCustomerPercentage",
    "TransactionCount"
]

customer_missing_by_country.to_csv(
    RESULTS_DIR / "customer_data_bias.csv"
)

print("\nCustomer data coverage:")
print(customer_missing_by_country.head(10))




print("\nModel performance differences:")
print(scenario_accuracy)



print("\nREQUIREMENT 6: FAIRNESS STRATEGIES")

fairness_strategies = {

    "Geographic representation": (
        "The retailer's transaction data may contain "
        "unequal country representation. Report "
        "country-specific findings rather than "
        "generalizing results to all consumers. "
        "The prompt dataset gives each included "
        "country the same number of country-level "
        "research examples."
    ),

    "Missing customer information": (
        "Retain transactions with missing customer IDs "
        "for overall sales analysis. Exclude them from "
        "analyses requiring customer identification. "
        "Report missing-data rates by country and "
        "avoid assuming missing records are random."
    ),

    "Prompt bias": (
        "Use neutral prompt wording. Avoid questions "
        "that assume a country, product or customer "
        "segment is superior without evidence. "
        "Require conclusions to use supplied data."
    ),

    "Unsupported competitor conclusions": (
        "Do not calculate market share without "
        "competitor sales and total market data. "
        "State that competitor comparisons cannot "
        "be determined from this dataset."
    ),

    "Evaluation fairness": (
        "Evaluate both models on identical questions "
        "with identical factual context. Report "
        "performance separately by research scenario "
        "to identify uneven results."
    ),

    "Limitations": (
        "The dataset does not provide sufficient "
        "demographic information to evaluate fairness "
        "across protected demographic groups. "
        "Geographic representation and scenario-level "
        "accuracy are not substitutes for a "
        "demographic fairness assessment."
    )

}


with open(
    RESULTS_DIR / "fairness_strategies.json",
    "w"
) as f:

    json.dump(
        fairness_strategies,
        f,
        indent=2
    )


for category, strategy in fairness_strategies.items():

    print(f"\n{category}:")
    print(strategy)




summary = {

    "dataset": "UCI Online Retail",

    "original_rows": int(len(df)),

    "cleaned_rows": int(len(sales)),

    "training_examples": len(training_examples),

    "evaluation_examples": len(evaluation_examples),

    "model": MODEL_NAME,

    "epochs": EPOCHS,

    "original_accuracy": float(original_accuracy),

    "fine_tuned_accuracy": float(fine_tuned_accuracy),

    "uk_transaction_percentage": float(uk_share),

    "missing_customer_percentage": float(
        missing_customer
    )

}


with open(
    RESULTS_DIR / "experiment_summary.json",
    "w"
) as f:

    json.dump(summary, f, indent=2)


print("\n================================")
print("SBA 928 EXPERIMENT COMPLETED")
print("================================")

print("\nResults saved to:")
print(RESULTS_DIR.resolve())
