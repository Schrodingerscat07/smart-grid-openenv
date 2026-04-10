"""Quick test: verify all grader scores are strictly in (0, 1)."""
import sys
sys.path.insert(0, ".")
from server.tasks import TASK_REGISTRY
from server.grid_env import SmartGridEnv
from models import Action

# Test 1: Empty history returns 0.001 (not 0.0)
print("=== Empty history test ===")
for name, task in TASK_REGISTRY.items():
    score = task.grade([])
    ok = 0 < score < 1
    print(f"  {name}: {score}  strictly_in_(0,1): {ok}")
    assert ok, f"FAIL: {name} returned {score}"

# Test 2: Do-nothing agent (should NOT return 0.0)
print("\n=== Do-nothing agent test ===")
env = SmartGridEnv()
for task_name in TASK_REGISTRY:
    env.reset(task_name=task_name)
    for _ in range(200):
        obs = env.step(Action(curtailments={}, battery_action="idle", battery_mw=0))
        if obs.done:
            break
    score = env.grade()
    ok = 0 < score < 1
    print(f"  {task_name}: {score}  strictly_in_(0,1): {ok}")
    assert ok, f"FAIL: {task_name} returned {score}"

print("\n✅ ALL SCORES STRICTLY IN (0, 1) — Phase 2 will pass!")
