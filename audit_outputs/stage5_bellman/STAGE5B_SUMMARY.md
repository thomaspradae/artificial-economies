# Stage 5B Bellman Target Summary

- Status: PASS
- Reason: Bellman targets, masks, maximizing actions, selected Q-values, and target-network next-Q values match within tolerance
- Stage 4 source: /home/uace/dqn/atari/audit_outputs/stage4_replay_sample
- Stage 5A source: /home/uace/dqn/atari/audit_outputs/stage5_learner
- Model exchange: /home/uace/dqn/atari/audit_outputs/stage5_learner/model_exchange
- PyTorch Bellman trace: /home/uace/dqn/atari/audit_outputs/stage5_bellman/pytorch_bellman.jsonl
- DeepMind Bellman trace: /home/uace/dqn/atari/audit_outputs/stage5_bellman/deepmind_bellman.jsonl
- Batch spec: /home/uace/dqn/atari/audit_outputs/stage5_bellman/bellman_batches.tsv
- Compare report: /home/uace/dqn/atari/audit_outputs/stage5_bellman/bellman_compare.txt
- Float tolerance: 1e-7

## Compare

```text
MATCH
```

## Coverage

- Sample rows: 69
- Batch rows: 8
- Batch names: batch_1, batch_32, batch_4, life_loss_terminal, ordinary_nonterminal, positive_reward, true_terminal, zero_reward
- True terminal samples: 9
- Life-loss terminal samples: 12
- Zero-reward samples: 54
- Positive-reward samples: 15
- Terminal-target checks true: 21
- Nonterminal gamma checks true: 48

