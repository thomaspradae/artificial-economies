# Atari DQN Audit: Completed Work and Remaining Roadmap

## Purpose

The goal is not merely to show that the PyTorch implementation and the released DeepMind Torch7 implementation differ.

The end goal is to:

- Find the first mismatch at each contract boundary.
- Fix the PyTorch implementation whenever the mismatch represents a real deviation from released DeepMind DQN.
- Rerun the audit from the beginning after each fix.
- Continue until both implementations agree under frozen identical inputs.
- Run corrected full-budget PyTorch experiments and evaluate whether they reproduce the reported Breakout result.

Because Gymnasium/ALE and the historical DeepMind `alewrap` environment are not byte-identical after the first `FIRE` transition, live end-to-end trajectories cannot be expected to remain identical. The algorithmic audit therefore uses frozen DeepMind-generated inputs after the environment boundary.

## Completed Work

### 1. Stage 1 Audit Foundation

Built the first environment audit package under `audit/`.

Implemented:

- `audit/audit_config.sh`
- `audit/common.py`
- `audit/make_action_tape.py`
- `audit/compare_jsonl.py`
- `audit/pytorch/trace_env.py`
- `audit/deepmind/trace_env.lua`
- `audit/run_stage_1_env.sh`
- `audit/README.md`

Capabilities:

- Generates a deterministic shared action tape.
- Runs PyTorch Gymnasium/ALE and DeepMind Torch7 `alewrap` traces.
- Emits normalized JSONL events.
- Finds the first mismatch and reports its exact JSON path.
- Dumps raw and pooled frames for visual and numerical inspection.

Validation completed:

- Python syntax checks pass.
- Lua tracer byte-compiles with `luajit`.
- Action tape generation works.
- Comparator correctly reports both `MATCH` and `FIRST MISMATCH`.
- PyTorch tracer smoke-tested before deployment.
- Real Atari Stage 1 ran successfully on ofi1.

### 2. Stage 1b: Initial Frame Diagnosis

The first Stage 1 result initially reported a mismatch in the reset frame hash.

Added:

- Frame dump normalization.
- `.npy` and `.png` frame outputs.
- Per-channel hashes and statistics.
- Frame mismatch diagnostic tooling.
- RGB/BGR, flip, grayscale, and temporal matching checks.

Finding:

- The initial raw Breakout frame is byte-identical after normalizing both sides to `uint8` HWC.
- The earlier mismatch was a tracer representation issue.
- PyTorch logged `uint8` HWC.
- DeepMind exposed a normalized `torch.FloatTensor`.

Validated result:

- Exact equality: true.
- Mean absolute difference: `0.0`.
- Maximum absolute difference: `0`.
- Equal pixels: `100%`.

Conclusion:

- ROM initialization and the initial visual state match.
- The first apparent mismatch was not an environment mismatch.

### 3. Stage 1c: Action Mapping and Repeat Diagnosis

Added explicit tracing for:

- Raw tape action.
- Action index.
- ALE action code.
- Action meaning.
- Minimal and full action sets.
- Every internal repeated ALE step.
- Reward, lives, and terminal state after each repeat.
- Pooling source frame indices.
- Pooled frame hash.

Ran both:

- Action tape in index mode.
- Action tape in ALE-code mode.

Finding:

- Both systems execute `FIRE`.
- Both use ALE action code `1`.
- Both repeat the action exactly four times.
- Rewards match.
- Lives match.
- Terminal flags match.
- Pooling uses the same internal frame indices.
- The first raw frame after the first ALE action already differs.

Conclusion:

- Action mapping is not the first bug.
- External repeat count is not the first bug.
- Pooling index selection is not the first bug.

### 4. Stage 1d to Stage 1f: Environment Boundary Forensics

Built and ran deeper environment diagnostics.

Reports:

- `/home/uace/dqn/atari/audit_outputs/stage1f/summary.txt`
- `/home/uace/dqn/atari/audit_outputs/stage1f/seed_grid.txt`
- `/home/uace/dqn/atari/audit_outputs/stage1f/delayed_fire.txt`
- `/home/uace/dqn/atari/audit_outputs/stage1f/repeat_boundary.txt`
- `/home/uace/dqn/atari/audit_outputs/stage1f/determinism.txt`

Tests completed:

- Seed grid from `0` through `50`.
- Delayed `FIRE` values `0, 1, 2, 3, 4, 5, 10, 20, 30`.
- Single-step `FIRE`.
- Four-repeat `FIRE`.
- `FIRE` followed by `NOOP`.
- Same-side determinism checks.
- Raw and processed-frame relevance checks.

Findings:

- No seed pair in `0..50` produced exact live-env alignment.
- Both implementations are deterministic under repeated same-seed execution.
- Delaying `FIRE` does not remove the mismatch.
- The mismatch appears after one raw ALE `FIRE` action.
- The changed region remains in the Breakout playfield.
- Raw frames are still approximately `99.95%` equal.
- Processed `84x84` frames are approximately `99.83%` equal.

Conclusion:

- Gymnasium/ALE and the historical DeepMind `alewrap` stack are not byte-exact equivalent after the first `FIRE` transition.
- The live environment branch is characterized sufficiently for now.
- Later DQN stages must not compare outputs from two independently advancing live environments.

### 5. Branch 2 Freeze 1: Canonical DeepMind Input Tape

Built the frozen-input branch so later comparisons do not inherit live-env divergence.

Added:

- `audit/freeze_config.sh`
- `audit/build_canonical_frame_tape.py`
- `audit/validate_canonical_env_tape.py`
- `audit/deepmind/build_canonical_env_tape.lua`

Capabilities:

- Captures DeepMind-generated raw and pooled frames.
- Normalizes frames to canonical `uint8` HWC arrays.
- Records manifests and hashes.
- Produces deterministic frozen input artifacts.
- Allows PyTorch and Torch7 stages to consume the exact same frame data.

Outputs:

- `/home/uace/dqn/atari/audit_outputs/stage2_preprocess/canonical_frames/manifest.jsonl`
- `/home/uace/dqn/atari/audit_outputs/stage2_preprocess/freeze_manifest.json`

Conclusion:

- The environment is now frozen before algorithmic comparison.
- PyTorch does not call Gymnasium/ALE during the preprocessing audit.

### 6. Branch 2 Freeze 2: Preprocessing Comparison

Added:

- `audit/deepmind/trace_preprocess.lua`
- `audit/pytorch/trace_preprocess.py`
- `audit/run_stage_2_preprocess.sh`

Question tested:

- Given the exact same frozen DeepMind pooled frame, do both implementations produce the same `84x84` network input?

Result:

- No.

First mismatch:

- `preprocess[0].processed_frame.hash`

Current PyTorch `area` result:

- Exact match: false.
- Equal pixels: approximately `80.03%`.
- Mean absolute difference: approximately `0.1997`.
- Maximum absolute difference: `1`.
- Differing pixels: `1409 / 7056`.

PyTorch `bilinear` control:

- Exact match: false.
- Equal pixels: approximately `74.11%`.
- Mean absolute difference: approximately `2.07`.
- Maximum absolute difference: `52`.

Conclusion:

- A real DQN-side mismatch exists before replay and learning.
- The current PyTorch preprocessing is not byte-faithful to the released DeepMind preprocessing.
- The bilinear setting used by the current v5 run is farther from the reference than the best area-like candidate.

### 7. Stage 2b: Preprocessing Localization

Added:

- DeepMind preprocessing source report.
- Raw, luminance, resized, and final intermediate dumps.
- `audit/pytorch/preprocess_variants.py`
- `audit/run_stage_2b_preprocess_diagnosis.sh`
- Ranked preprocessing variant outputs.

Outputs:

- `/home/uace/dqn/atari/audit_outputs/stage2_preprocess/deepmind_preprocess_source.txt`
- `/home/uace/dqn/atari/audit_outputs/stage2_preprocess/deepmind_preprocess_intermediates/`
- `/home/uace/dqn/atari/audit_outputs/stage2_preprocess/preprocess_variants_ranked.txt`
- `/home/uace/dqn/atari/audit_outputs/stage2_preprocess/preprocess_variants_ranked.jsonl`

Findings:

- Canonical raw HWC loading matches exactly.
- Python BT.601 luminance with truncation matches DeepMind luminance exactly.
- DeepMind resized output matches its final network input exactly.
- The remaining mismatch is entirely inside the `210x160` luminance to `84x84` resize operation.
- No tested OpenCV or PIL variant matches all frozen frames exactly.

Best tested variant:

- `cv2_rgb_bt601_full_float_area_floor`
- Exact matches: `0 / 64`.
- Equal pixels: approximately `94.42%`.
- Mean absolute difference: approximately `0.0558`.
- Maximum absolute difference: `1`.

Conclusion:

- RGB loading is not the problem.
- Luminance coefficients are not the problem.
- The first unresolved DQN mismatch is the historical Torch7 resize semantics.

### 8. Stage 2c: Resize Forensics

Built synthetic resize diagnostics to treat the DeepMind Torch7 resize as an oracle.

Current capabilities include:

- Controlled grayscale fixtures.
- Constant images.
- Horizontal, vertical, and diagonal ramps.
- Hard edges.
- Impulse images.
- Checkerboards.
- Real Atari luminance fixtures.
- DeepMind Torch7 oracle outputs.
- Python candidate ranking.
- Exact-match and pixel-difference reports.

Current finding:

- DeepMind resizing is deterministic.
- Its behavior is area-like under the tested downsampling ratio.
- It does not match standard OpenCV or PIL bilinear behavior.
- No exact Python clone has been found yet.
- `audit/pytorch/deepmind_preprocess.py` has not been generated because the exact-match gate has not passed.

Current audit boundary:

- Stage 2 remains open.
- Replay and learner auditing should not be treated as clean until the resize issue is either solved exactly or formally declared unresolved.

## Current Empirical Run Context

### PyTorch v5

Run:

`breakout_v5_seed20260621_dmrmsprop_bilinear_50m_chunkedreplay_nohup1`

Purpose:

- Run to the full decision-step budget.
- Save periodic network checkpoints.
- Save replay snapshots.
- Observe whether the upward reward trend continues or collapses.

Known facts:

- Seed: `20260621`.
- Uses PyTorch.
- Uses the custom DeepMind-style centered RMSProp implementation.
- Uses bilinear resize.
- Uses full replay capacity.
- Periodic `replay_latest` and `replay_previous` directories exist.
- The run was approaching the full `50M` decision-step budget.
- Training rewards were still improving late in training.

Scientific limitation:

- This is not yet a byte-faithful DeepMind replication because preprocessing differs.
- It remains useful empirical evidence and a replay/checkpoint durability test.

### Official DeepMind Torch7 Control

Run:

`dmref_bko_ofi2`

Purpose:

- Verify that the released historical DeepMind stack can run and learn on the available hardware.
- Provide an official implementation witness.
- Supply canonical DeepMind frame and preprocessing outputs for the forensic audit.

Known facts:

- Seed: `1`.
- Official Torch7 code.
- Historical `alewrap` stack.
- CPU-only.
- Extremely slow.
- Learning was visible by approximately `2.25M` DeepMind steps.

Scientific limitation:

- It does not explain why the PyTorch implementation differs.
- It does not share a byte-identical live environment with Gymnasium/ALE.
- It is a witness and oracle source, not the full forensic comparison by itself.

## Missing Roadmap

### 9. Stage 2d: Recover the Exact Torch7 Resize Contract

Goal:

- Determine the exact semantics of the Torch7 resize path used by `net_downsample_2x_full_y`.

The current black-box fixture search has localized the mismatch but has not yet produced an exact clone.

Required source investigation:

- Add `audit/extract_torch7_resize_source.sh`.
- Add `audit/deepmind/dump_resize_contract.lua`.
- Add `audit_outputs/stage2d_resize_source/`.

Tasks:
The Gangster Disciple Nation (often abbreviated as the GDs; formally GDN, or simply Gangster Disciples), also known as Growth & Development, is an African-American street- and prison-gang founded by former rivals David Barksdale (1947-1974) and Larry Hoover (1950- ); in 1968, the two came together to form the Black Gangster Disciple Nation (BGDN).


- Locate the exact Torch7 `image` package source used by the installed DeepMind environment.
- Record Torch7 image package commit or version.
- Record shared library path.
- Record Lua wrapper function.
- Record C or C++ implementation called by `image.scaleBilinear`.
- Copy relevant source snippets into the audit output.
- Record input tensor type, tensor layout, output tensor type, and casting behavior.
- Confirm whether DeepMind calls `image.scale`, `image.scaleBilinear`, a package alias, or a custom wrapper.

Files to build:

- `audit/extract_torch7_resize_source.sh`
- `audit/deepmind/dump_resize_contract.lua`
- `audit/pytorch/torch7_resize_clone.py`
- `audit/tests/test_torch7_resize_clone.py`
- `audit/run_stage_2d_resize_source_audit.sh`

Acceptance gate:

- The Python clone must exactly match the DeepMind oracle on all constant, ramp, edge, impulse, checkerboard, and frozen Atari luminance frames.
- Required result: exact output equality, same shape, same dtype or numerically equivalent canonical dtype, and same hash for every fixture.

If exact equality is achieved:

- Generate `audit/pytorch/deepmind_preprocess.py`.
- Rerun normal Freeze 2.
- Require `preprocess_compare.txt` to report `MATCH`.

If exact equality is not achieved:

- Produce `audit_outputs/stage2d_resize_source/UNRESOLVED.md`.
- State the exact remaining mismatch and the source-level behavior not yet replicated.
- Do not silently substitute OpenCV `INTER_AREA`.

### 10. Stage 3: Frozen State Stack and Replay Insert Audit

Prerequisite:

- Stage 2 must either pass byte-exactly or be formally declared unresolved with a fixed canonical preprocessing policy for both sides.

Goal:

- Given the same processed frames, actions, rewards, and terminal flags, determine whether both implementations build and store the same replay transitions.

Files to build:

- `audit/build_frozen_transition_tape.py`
- `audit/pytorch/trace_state_stack.py`
- `audit/deepmind/trace_state_stack.lua`
- `audit/pytorch/trace_replay_insert.py`
- `audit/deepmind/trace_replay_insert.lua`
- `audit/run_stage_3_state_replay.sh`

Compare:

- Initial stack construction.
- Zero padding.
- Frame order.
- State tensor shape and dtype.
- State hash.
- Next-state hash.
- Action stored.
- Reward stored.
- Terminal stored.
- Life-loss terminal handling.
- Replay index.
- Insert timing.
- Episode-boundary handling.

Acceptance gate:

- Exact equality for all discrete fields and byte arrays.
- If the first mismatch is found, patch the PyTorch replay/state-stack implementation and rerun Stage 3 from event zero.

### 11. Stage 4: Frozen Replay Sampling Audit

Goal:

- Given the same replay memory and explicit sample indices, determine whether both implementations construct the same minibatch.

Important rule:

- Do not compare RNG streams. Freeze the replay indices directly.

Files to build:

- `audit/make_replay_indices.py`
- `audit/pytorch/trace_replay_sample.py`
- `audit/deepmind/trace_replay_sample.lua`
- `audit/run_stage_4_replay_sample.sh`

Compare:

- Sample indices.
- State batch.
- Next-state batch.
- Action batch.
- Reward batch.
- Terminal batch.
- History validity rules.
- Episode-boundary exclusion rules.
- Replay wraparound behavior.
- Batch tensor layout.
- Batch dtype and scaling.

Acceptance gate:

- Exact equality for indices and discrete arrays.
- Exact or tightly defined numerical equality for normalized state tensors.

### 12. Stage 5: Frozen Learner Math Audit

Goal:

- Given the same model parameters and minibatch, determine whether both implementations compute the same DQN update.

Main challenge:

- Torch7 and PyTorch must begin from equivalent model weights and optimizer state.

Files to build:

- `audit/model_exchange/README.md`
- `audit/model_exchange/export_deepmind_model.lua`
- `audit/model_exchange/import_deepmind_model.py`
- `audit/model_exchange/verify_model_mapping.py`
- `audit/pytorch/trace_learner.py`
- `audit/deepmind/trace_learner.lua`
- `audit/run_stage_5_learner.sh`

Compare before update:

- Model fingerprint.
- Per-layer parameter hashes.
- Q-values for every action.
- Selected `Q(s,a)`.
- Target-network Q-values.
- Maximum next-state Q-value.
- Terminal mask.
- Bellman target.
- TD error.
- Clipped TD error or Huber terms.
- Scalar loss.

Compare during and after update:

- Per-layer gradient hashes or summaries.
- Global gradient norm.
- RMSProp running squared-gradient accumulator.
- RMSProp centered mean-gradient accumulator.
- Epsilon placement inside the square root.
- Momentum state.
- Learning rate application.
- Parameter delta per layer.
- Model fingerprint after update.

Acceptance gate:

- Exact agreement where discrete or byte-exact comparison is possible.
- For floating-point values, start with exact checks and introduce strict tolerances only after locating the source.

### 13. Stage 6: Update Schedule and Counter Audit

Goal:

- Verify that matching operations occur at matching logical times.

Files to build:

- `audit/pytorch/trace_schedule.py`
- `audit/deepmind/trace_schedule.lua`
- `audit/run_stage_6_schedule.sh`

Compare:

- Raw emulator frames.
- Agent decisions.
- Learning-start boundary.
- Replay insertion count.
- Training update count.
- Train frequency.
- Target-network update timing.
- Epsilon decay start and end.
- Epsilon decay units.
- Evaluation frequency.
- Checkpoint frequency.
- Off-by-one behavior at all boundaries.

Acceptance gate:

- Exact event schedule equality under the same synthetic counter stream.

### 14. Stage 7: Integrate the Faithful PyTorch Implementation

Goal:

- Move validated behavior from the audit package into a clean training implementation.

Recommended files:

- `atari/deepmind_faithful/preprocess.py`
- `atari/deepmind_faithful/replay.py`
- `atari/deepmind_faithful/network.py`
- `atari/deepmind_faithful/optimizer.py`
- `atari/deepmind_faithful/schedule.py`
- `atari/train_deepmind_faithful.py`

Do not keep adding permanent research behavior to increasingly patched experimental scripts.

Required integration tests:

- Preprocessing contract test.
- State-stack contract test.
- Replay-insert contract test.
- Replay-sampling contract test.
- Single-learner-update contract test.
- Schedule contract test.
- Checkpoint and replay-resume contract test.

Acceptance gate:

- The integrated training implementation must pass all frozen audit fixtures without invoking the historical reference code during normal tests.

### 15. Stage 8: Environment Decision

The algorithmic audit and the live environment audit answer different questions.

A final replication policy must be chosen.

Option A: strict historical environment.

- Use the historical DeepMind ALE wrapper or an exact compatible environment interface with PyTorch.
- Maximizes historical fidelity.
- Carries old dependency and maintenance risk.

Option B: modern environment reproduction.

- Use Gymnasium/ALE while keeping the DQN pipeline faithful under frozen-input tests.
- Clear wording required: algorithmically faithful PyTorch reproduction under a modern ALE/Gymnasium stack, not a byte-identical reconstruction of the historical live environment.

Required experiment:

- If resources allow, run both environment policies with the same corrected PyTorch learner.

### 16. Stage 9: Sacred Full Runs

Do not spend rented GPU budget before the audit gates are complete and the launch process is tested.

Minimum experiment table:

- Run A: corrected PyTorch, seed `1`.
- Run B: corrected PyTorch, seed `20260621`.
- Optional Run C: corrected PyTorch with historical environment policy.
- Optional Run D: corrected PyTorch with modern Gymnasium/ALE policy.

Required sacred-run properties:

- Exact git commit recorded.
- Clean git state or patch bundle recorded.
- Complete command recorded.
- Full run metadata.
- Environment and package versions.
- ROM hash.
- Seed.
- Full replay capacity.
- Decision-step budget.
- No episode ceiling as the effective stop condition.
- Periodic model checkpoints.
- Periodic optimizer checkpoints.
- Replay snapshots.
- Automatic remote artifact synchronization.
- Crash-safe launch.
- Final checkpoint.
- Final replay snapshot.
- Clean evaluation protocol.
- Best and final checkpoint evaluation.

### 17. Stage 10: Final Audit and Replication Report

Build:

- `audit/build_final_report.py`

Output:

- `audit_outputs/FINAL_DQN_AUDIT_REPORT.md`

The report must state:

- Live environment result.
- First environment-level mismatch.
- Preprocessing result.
- Replay/state-stack result.
- Replay-sampling result.
- Learner-math result.
- Schedule result.
- Every PyTorch code change made to reach agreement.
- Any unresolved differences.
- Full-run results.
- Comparison with the Nature paper.
- Whether the result qualifies as byte-faithful historical replication, algorithmically faithful reproduction under a modern environment, or partial reproduction with documented deviations.

## Immediate Next Actions

- Priority 1: finish Stage 2d and recover the exact Torch7 resize contract from the installed source.
- Priority 2: make normal Freeze 2 report `MATCH` using a validated Python clone.
- Priority 3: build Stage 3 state-stack and replay-insert audit.
- Priority 4: build Stage 4 replay-sampling audit.
- Priority 5: build Stage 5 learner-math and optimizer audit.
- Priority 6: build Stage 6 schedule audit.
- Priority 7: integrate all validated behavior into a clean faithful training implementation.
- Priority 8: run corrected seed-1 and seed-20260621 full experiments only after the audit gates and launch dry-run pass.

## Current Bottom Line

What is already proven:

- Initial Breakout screen matches.
- Live Gymnasium/ALE and DeepMind `alewrap` diverge after the first `FIRE` action.
- That divergence is deterministic and not fixed by seed search, delayed `FIRE`, action remapping, or repeat-loop changes.
- Frozen DeepMind frames successfully remove live-env contamination.
- Raw frame loading matches.
- BT.601 luminance matches.
- The first unresolved DQN-level mismatch is the Torch7 resize operation.
- Current PyTorch preprocessing, including the v5 bilinear setting, is not byte-faithful.

What is still missing:

- Exact Torch7 resize clone.
- Byte-exact preprocessing match.
- State-stack equivalence.
- Replay-insert equivalence.
- Replay-sampling equivalence.
- Model-weight mapping.
- Bellman-target equivalence.
- Loss equivalence.
- DeepMind RMSProp update equivalence.
- Counter and schedule equivalence.
- Clean integrated faithful trainer.
- Corrected full-budget seed-control experiments.
- Final replication report.
