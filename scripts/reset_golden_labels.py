import json
from pathlib import Path


GOLDEN_FILE = Path("data/golden/golden_set.jsonl")


def main():
    if not GOLDEN_FILE.exists():
        print(f"File not found: {GOLDEN_FILE}")
        return

    records = []

    with GOLDEN_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            record["gold_intent"] = ""
            record["gold_action"] = ""
            record["gold_escalation_reason"] = ""
            record["annotator_notes"] = ""

            records.append(record)

    with GOLDEN_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print("=" * 60)
    print("GOLDEN SET LABELS RESET")
    print("=" * 60)
    print(f"Total examples : {len(records)}")
    print(f"File           : {GOLDEN_FILE}")
    print()
    print("All gold labels have been cleared.")
    print("You can now manually label the examples.")


if __name__ == "__main__":
    main()