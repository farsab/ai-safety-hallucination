import argparse
from src.pipeline import SafetyPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str, help="User query")
    args = parser.parse_args()

    pipe = SafetyPipeline()
    result = pipe.run(args.query)

    print("\n=== Answer ===\n")
    print(result["answer"])
    if result.get("citations"):
        print("\n=== Citations ===")
        for i, c in enumerate(result["citations"], 1):
            print(f"[{i}] {c['source']}")

    if result.get("reasons"):
        print("\n=== Safety Checks ===")
        for k, v in result["reasons"].items():
            print(f"- {k}: {v}")

if __name__ == "__main__":
    main()
