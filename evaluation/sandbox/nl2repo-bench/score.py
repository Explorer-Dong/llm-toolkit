import json
from pathlib import Path

score = 0.0
task_num = 0
for res_file in Path("./result").glob("*.json"):
    task_num += 1
    with open(res_file, encoding="utf-8") as f:
        data = json.load(f)
        score += data["post_process_result"]["pytest_results"]["success_rate"]
print(f"task number: {task_num}, avg. success rate: {score / task_num * 100:.2f} %")
