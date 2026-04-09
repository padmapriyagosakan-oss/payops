"""Test grader property and grade_episode changes."""
import sys
sys.path.insert(0, '.')
from payops_env.tasks import TASKS
from payops_env.grader import grade_episode

# Test 1: task.grader property
print('=== Test 1: task.grader property ===')
missing = [t.task_id for t in TASKS if not hasattr(t, 'grader')]
if missing:
    print(f'FAIL: tasks missing grader property: {missing}')
    sys.exit(1)
else:
    print(f'PASS: all {len(TASKS)} tasks have grader property')

# Spot-check grader content
t0 = TASKS[0]
g = t0.grader
assert 'type' in g and g['type'] == 'action_match', f'grader bad: {g}'
assert 'correct_action' in g
assert 'partial_credit' in g
assert 'requires_investigation' in g
assert 'regulatory_action' in g
assert 'key_flags' in g
print(f'PASS: grader property has required keys: {sorted(g.keys())}')

# Test 2: grade_episode per_task_rewards have grader key
print()
print('=== Test 2: grade_episode per_task_rewards have grader key ===')
sample_tasks = list(TASKS[:5])
sample_actions = [t.correct_action for t in sample_tasks]
result = grade_episode(sample_actions, sample_tasks)
missing_gr = [pt['task_id'] for pt in result.per_task_rewards if 'grader' not in pt]
if missing_gr:
    print(f'FAIL: per_task_rewards entries missing grader key: {missing_gr}')
    sys.exit(1)
else:
    print(f'PASS: all {len(result.per_task_rewards)} per_task_rewards entries have grader key')
print(f'PASS: score={result.normalised_score}')

# Test all 30 tasks
result_all = grade_episode([t.correct_action for t in TASKS], list(TASKS))
missing_all = [pt['task_id'] for pt in result_all.per_task_rewards if 'grader' not in pt]
assert not missing_all, f'FAIL: {missing_all}'
print(f'PASS: all 30 tasks graded with grader key in per_task_rewards')
print(f'PASS: score={result_all.normalised_score}')

# Test 3: openenv.yaml has task definitions with grader
print()
print('=== Test 3: openenv.yaml task definitions ===')
import yaml
with open('openenv.yaml') as f:
    d = yaml.safe_load(f)
# tasks: is now a flat list (changed from dict+definitions to list for platform compat)
tasks_section = d.get('tasks', [])
if isinstance(tasks_section, list):
    defs = tasks_section
else:
    defs = tasks_section.get('definitions', [])
tasks_with_grader = [t for t in defs if 'grader' in t]
print(f'PASS: openenv.yaml has {len(defs)} task definitions, {len(tasks_with_grader)} with grader')
assert len(tasks_with_grader) >= 3, f'FAIL: only {len(tasks_with_grader)} tasks with grader'

print()
print('ALL TESTS PASSED')
