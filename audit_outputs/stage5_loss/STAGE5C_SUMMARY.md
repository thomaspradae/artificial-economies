# Stage 5C TD Error Clipping / Loss Semantics Summary

- Status: PASS
- Reason: raw TD errors, clipping, sparse selected-action output gradients, and threshold controls match DeepMind
- Stage 5B source: /home/uace/dqn/atari/audit_outputs/stage5_bellman
- Fixture: /home/uace/dqn/atari/audit_outputs/stage5_loss/loss_fixture.json
- Threshold controls: /home/uace/dqn/atari/audit_outputs/stage5_loss/loss_threshold_controls.tsv
- DeepMind source report: /home/uace/dqn/atari/audit_outputs/stage5_loss/deepmind_loss_source.txt
- PyTorch loss trace: /home/uace/dqn/atari/audit_outputs/stage5_loss/pytorch_loss.jsonl
- DeepMind loss trace: /home/uace/dqn/atari/audit_outputs/stage5_loss/deepmind_loss.jsonl
- Compare report: /home/uace/dqn/atari/audit_outputs/stage5_loss/loss_compare.txt
- Regression test report: /home/uace/dqn/atari/audit_outputs/stage5_loss/loss_test.txt
- Float tolerance: 1e-7

## Source Findings

## Findings

- Raw TD error convention: `target - Q(s,a)`.
- `target` is `r + gamma * (1 - terminal) * max_a Q_target(s2,a)` before clipping.
- `clip_delta=1` clamps the TD error in place to `[-1, 1]`.
- No explicit Huber scalar criterion is computed in the released learner path.
- The clamped TD error is written into a sparse `targets` tensor with nonzero value only at the selected action.
- `network:backward(s, targets)` receives that sparse clipped-delta tensor directly.
- There is no division by minibatch size in `getQUpdate` or before `network:backward`.
- The optimizer update is outside Stage 5C and is not executed by this runner.

## Compare

```text
MATCH
```

## Regression Test

```text
test_deepmind_no_batch_mean_reduction (audit.tests.test_loss_match.LossSemanticsMatchTest.test_deepmind_no_batch_mean_reduction) ... ok
test_pytorch_and_deepmind_effective_rows_match (audit.tests.test_loss_match.LossSemanticsMatchTest.test_pytorch_and_deepmind_effective_rows_match) ... ok
test_required_batches_present (audit.tests.test_loss_match.LossSemanticsMatchTest.test_required_batches_present) ... ok
test_sparse_selected_action_output_gradient (audit.tests.test_loss_match.LossSemanticsMatchTest.test_sparse_selected_action_output_gradient) ... ok
test_stage5c_compare_passed (audit.tests.test_loss_match.LossSemanticsMatchTest.test_stage5c_compare_passed) ... ok
test_threshold_clipping_contract (audit.tests.test_loss_match.LossSemanticsMatchTest.test_threshold_clipping_contract) ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.012s

OK
```

## Coverage

- Sample rows: 153
- Batch rows: 27
- Batch names: batch_1, batch_32, batch_4, life_loss_terminal, ordinary_nonterminal, positive_reward, real_single_0_1, real_single_1_2, real_single_2_3, real_single_3_4, threshold_all, threshold_batch_32, threshold_duplicated, threshold_mixed_4, threshold_single_0p0, threshold_single_0p5, threshold_single_0p999, threshold_single_1p0, threshold_single_1p001, threshold_single_2p0, threshold_single_m0p5, threshold_single_m0p999, threshold_single_m1p0, threshold_single_m1p001, threshold_single_m2p0, true_terminal, zero_reward
- Threshold-control samples: 80
- Clipped samples: 40
- At-threshold samples: 14
- Unselected-action-zero checks true: 153

