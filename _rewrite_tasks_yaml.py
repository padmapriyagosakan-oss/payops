"""
Rewrite the tasks section of openenv.yaml to use Python callable grader paths.
The platform expects: grader: payops_env.server.graders:EASY001Grader
instead of an inline YAML dict.
"""
import sys
sys.path.insert(0, ".")
from payops_env.tasks import TASKS

with open("openenv.yaml") as f:
    content = f.read()

# Build new tasks block — one entry per task with string grader path
lines = []
lines.append("# tasks: grader field points to a Python callable (module:Class format)")
lines.append("tasks:")
for t in TASKS:
    class_name = t.task_id.replace("-", "") + "Grader"
    lines.append(f"  - id: {t.task_id}")
    lines.append(f"    name: \"{t.task_id}\"")
    lines.append(f"    difficulty: {t.difficulty}")
    lines.append(f"    grader: payops_env.server.graders:{class_name}")

new_tasks_block = "\n".join(lines)

# Verify counts
assert new_tasks_block.count("  - id:") == 30, "Expected 30 task entries"
assert new_tasks_block.count("    grader: payops_env.server.graders:") == 30, "Expected 30 grader paths"
print("Block verification: 30 tasks, 30 string grader paths - OK")

# Replace in openenv.yaml: from "# tasks:" comment to just before "# ── Budget"
old_start = "# tasks: flat list — platform iterates this directly for grader check"
budget_marker = "\n# ── Budget"

if old_start not in content:
    print("ERROR: Could not find tasks section start marker")
    sys.exit(1)

tasks_start_pos = content.index(old_start)
budget_start_pos = content.index(budget_marker)

new_content = content[:tasks_start_pos] + new_tasks_block + "\n" + content[budget_start_pos:]

with open("openenv.yaml", "w") as f:
    f.write(new_content)

print("openenv.yaml rewritten successfully")

# Parse and verify
import yaml
with open("openenv.yaml") as f:
    d = yaml.safe_load(f.read())

tasks = d.get("tasks", [])
print(f"Parsed tasks: type={type(tasks).__name__}, count={len(tasks)}")
assert isinstance(tasks, list) and len(tasks) == 30

grader_strings = [t for t in tasks if isinstance(t.get("grader"), str)]
print(f"Tasks with string grader: {len(grader_strings)}/30")
assert len(grader_strings) == 30, "All graders must be strings"

# Spot-check format
g = tasks[0]["grader"]
assert "payops_env.server.graders:" in g, f"Bad grader format: {g}"
assert g.endswith("Grader"), f"Grader should end with Grader: {g}"
print(f"First task grader: {g}")
print("VERIFICATION PASSED")
