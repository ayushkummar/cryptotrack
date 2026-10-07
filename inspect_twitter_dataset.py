from datasets import load_dataset

print("Loading dataset metadata...")

ds = load_dataset(
    "StephanAkkerman/crypto-stock-tweets",
    split="train",
    streaming=True
)

print("\nDataset loaded in streaming mode.")
print("Columns:")
print(ds.features)

print("\nFirst 5 rows:")
for i, row in enumerate(ds):
    print(row)
    if i == 4:
        break