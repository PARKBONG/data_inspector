import json
from pathlib import Path

jsonl_path = Path("DB/260227/260227_155409/IR_High/0.jsonl")

print("File exists: %s" % jsonl_path.exists())
if jsonl_path.exists():
    print("File size: %d bytes" % jsonl_path.stat().st_size)
    print("\n=== First 3 lines ===")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            try:
                data = json.loads(line)
                print("\nLine %d:" % (i+1))
                print("  Keys: %s" % list(data.keys()))
                print("  Time: %s" % data.get('Time'))
                print("  MaxTemp: %s" % data.get('MaxTemp'))
                print("  PeakX: %s" % data.get('PeakX'))
                print("  PeakY: %s" % data.get('PeakY'))
                print("  ContourPoints type: %s" % type(data.get('ContourPoints')))
                cp = data.get('ContourPoints', [])
                print("  ContourPoints length: %d" % len(cp))
                if cp:
                    print("  ContourPoints sample: %s" % cp[:2])
            except Exception as e:
                print("Error on line %d: %s" % (i+1, str(e)))
