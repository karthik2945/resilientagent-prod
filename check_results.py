import json

d = json.load(open("baseline/checkpoints/training_results.json"))
print(f"Training time: {d['training_time_seconds']}s")
print(f"Best avg score: {d['best_avg_score']}")
print(f"Final eval: {d['final_eval']}")
print(f"Final epsilon: {d['final_epsilon']}")
r = d["reward_history"]
print(f"Reward trend: first 10 avg = {sum(r[:10])/10:.3f}, last 10 avg = {sum(r[-10:])/10:.3f}")
ls = d["loss_history"]
nz = [x for x in ls if x > 0]
print(f"Loss trend: first 10 nonzero avg = {sum(nz[:10])/10:.6f}, last 10 avg = {sum(nz[-10:])/10:.6f}")
