"""
CLI entrypoint for synthetic data generation.
Usage: python generate.py --type tabular --rows 5000
"""
import argparse, json
from pathlib import Path
from generators.tabular_generator import generate_synthetic_customer_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["tabular","text"], default="tabular")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--output", default="results/synthetic_data.csv")
    args = parser.parse_args()
    Path("results").mkdir(exist_ok=True)

    if args.type == "tabular":
        print(f"Generating {args.rows} synthetic customer records...")
        df = generate_synthetic_customer_data(args.rows)
        df.to_csv(args.output, index=False)
        print(f"✓ Saved to {args.output}")
        print(df.describe().round(2).to_string())
    else:
        print("Text generation requires ANTHROPIC_API_KEY — see generators/llm_synthesizer.py")

if __name__ == "__main__":
    main()
