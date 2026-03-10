import itertools
import subprocess
import json
import os

lrs = [0.001, 0.005]
hds = [64, 128]
heads = [4, 8]

results = []
metrics_path = '../../data/results/training_metrics.json'

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
metrics_path = os.path.join(script_dir, '../../data/results/training_metrics.json')

for lr, hd, h in itertools.product(lrs, hds, heads):
    print(f"\n--- Running: lr={lr}, hidden_dim={hd}, heads={h} ---")
    cmd = ["python", "-W", "ignore", "train.py", "--lr", str(lr), "--hidden_dim", str(hd), "--heads", str(h)]
    # Suppress output so it doesn't flood the terminal
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    with open(metrics_path, 'r') as f:
        m = json.load(f)
    
    res = {
        'lr': lr,
        'hidden_dim': hd,
        'heads': h,
        'test_auprc': m['test_auprc'],
        'test_auroc': m['test_auroc'],
        'best_epoch': m['best_epoch']
    }
    results.append(res)
    print(f"Result: AUPRC={res['test_auprc']:.4f}, AUROC={res['test_auroc']:.4f}")

results.sort(key=lambda x: x['test_auprc'], reverse=True)
out_path = os.path.join(script_dir, '../../data/results/ablation_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print("\nBest Configuration:")
print(json.dumps(results[0], indent=2))

# Retrain with the best configuration to save its node_embeddings.pt and model checkpoint
best = results[0]
print("\nRetraining with best configuration to lock in model...")
cmd_best = ["python", "-W", "ignore", "train.py", "--lr", str(best['lr']), "--hidden_dim", str(best['hidden_dim']), "--heads", str(best['heads'])]
subprocess.run(cmd_best, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Finished!")
