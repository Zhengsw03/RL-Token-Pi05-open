import importlib.util
import io
import json
import pickle
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def _load_stage2_module():
    script_path = Path(__file__).parents[1] / "scripts" / "train_rlt_stage2_pi05.py"
    spec = importlib.util.spec_from_file_location("train_rlt_stage2_pi05", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pi05ActionScaleTest(unittest.TestCase):
    def test_subsampled_pi05_actions_stay_in_model_normalized_space(self):
        module = _load_stage2_module()
        actions = torch.tensor([[[0.25, -0.75], [0.50, -0.50], [0.75, -0.25]]])

        result = module.subsample_pi05_actions(actions, action_dim=2, stride=2, num_steps=2)

        torch.testing.assert_close(result, torch.tensor([[[0.25, -0.75], [0.75, -0.25]]]))

    def test_action_cache_takes_a_full_execution_horizon(self):
        module = _load_stage2_module()
        cache = module.Pi05ActionCache()
        actions = torch.tensor([[[0.1, 9.0], [0.2, 8.0], [0.3, 7.0], [0.4, 6.0]]])
        embedding = torch.tensor([[4.0]])
        cache.refresh(actions, embedding, action_dim=1)

        result, start_index, result_embedding = cache.take(2)

        torch.testing.assert_close(result, torch.tensor([[[0.1], [0.2]]]))
        self.assertEqual(start_index, 0)
        self.assertIs(result_embedding, embedding)
        torch.testing.assert_close(cache.remaining_raw_actions(), torch.tensor([[[0.3, 7.0], [0.4, 6.0]]]))
        self.assertTrue(cache.exhausted() is False)

    def test_action_cache_rejects_partial_execution_horizon(self):
        module = _load_stage2_module()
        cache = module.Pi05ActionCache()
        cache.refresh(torch.zeros(1, 3, 1), torch.zeros(1, 1), action_dim=1)

        with self.assertRaises(ValueError):
            cache.take(4)

    def test_fixed_std_sample_is_stochastic(self):
        module = _load_stage2_module()
        torch.manual_seed(7)
        mean = torch.zeros(1, 2, 1)
        std = torch.full_like(mean, 0.25)

        sampled = module.sample_fixed_std_action(mean, std)

        self.assertFalse(torch.equal(sampled, mean))
        torch.manual_seed(7)
        torch.testing.assert_close(sampled, module.sample_fixed_std_action(mean, std))

    def test_polyak_update_moves_target_toward_source(self):
        module = _load_stage2_module()
        source = torch.nn.Linear(1, 1, bias=False)
        target = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            source.weight.fill_(4.0)
            target.weight.fill_(0.0)

        module.polyak_update(source, target, 0.25)

        torch.testing.assert_close(target.weight, torch.tensor([[1.0]]))

    def test_teleop_motion_does_not_request_takeover_key_only(self):
        # Intervention is now triggered ONLY by an explicit key (h); leader motion
        # must not set a takeover request (it only drives the HIL stillness/resume).
        module = _load_stage2_module()
        manager = module.TeleopManager(None, threshold=0.5, trigger_frames=1, release_frames=2)
        base = {f"{joint}.pos": 0.0 for joint in manager.JOINT_NAMES}

        for joint in manager.JOINT_NAMES:
            manager._update_motion_locked(base)
            moved = dict(base)
            moved[f"{joint}.pos"] = 0.6
            manager._update_motion_locked(moved)
            self.assertFalse(manager.consume_takeover_request(), joint)
            manager.reset_episode()

    def test_teleop_stillness_never_auto_resumes_manual_exit_only(self):
        # HIL exit is manual only ('h'/'s' keys): leader stillness must never
        # produce a resume request.
        module = _load_stage2_module()
        manager = module.TeleopManager(None, threshold=0.5, trigger_frames=1, release_frames=2)
        action = {f"{joint}.pos": 0.0 for joint in manager.JOINT_NAMES}
        manager.enter_human_mode(0.0)
        for _ in range(50):  # long stillness after human control
            manager._update_motion_locked(action)
        self.assertFalse(manager.consume_resume_request())

    def test_executed_action_stack_preserves_time_then_action_dimensions(self):
        sent_actions = [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])]

        result = torch.stack(sent_actions, dim=0).unsqueeze(0)

        self.assertEqual(result.shape, (1, 2, 2))
        torch.testing.assert_close(result, torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]))

        module = _load_stage2_module()
        actions = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]])

        result = module.make_pi05_reference_chunk(
            actions,
            action_dim=1,
            start_index=1,
            stride=2,
            num_steps=3,
        )

        torch.testing.assert_close(result, torch.tensor([[[1.0], [3.0], [5.0]]]))


class Stage2SafetyContractTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_stage2_module()

    def test_relative_target_limit_can_be_explicitly_disabled(self):
        self.assertIsNone(self.module.parse_max_relative_target(None))

    def test_dry_run_output_preflight_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new-output"
            self.module.prepare_output_dir(output, dry_run=True)
            self.assertFalse(output.exists())

    def test_output_preflight_rejects_existing_contents(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            (output / "prior.pt").write_bytes(b"prior")
            with self.assertRaises(FileExistsError):
                self.module.prepare_output_dir(output, dry_run=False)

    def test_tokenizer_resolves_from_checkpoint_processor_without_gemma_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root)
            (checkpoint / "policy_preprocessor.json").write_text(json.dumps({
                "steps": [{
                    "registry_name": "tokenizer_processor",
                    "config": {"tokenizer_name": "local/paligemma"},
                }]
            }))
            self.assertEqual(
                self.module.resolve_tokenizer_path(checkpoint, ""),
                "local/paligemma",
            )

    def test_prompt_discretization_matches_pi05_processor_below_range(self):
        normalized = torch.tensor([-1.1, -1.0, 0.0, 1.0])
        self.assertEqual(self.module.discretize_state(normalized).split()[0], "-1")

    def test_quantile_contract_rejects_non_six_dimensional_stats(self):
        with self.assertRaisesRegex(ValueError, "exactly 6D"):
            self.module._validate_quantiles({
                "observation.state": {"q01": [0] * 5, "q99": [1] * 5},
                "action": {"q01": [0] * 6, "q99": [1] * 6},
            })

    def test_mapped_bad_camera_frame_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported frame type"):
            self.module.validate_camera_frame(object(), "wrist")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.module.validate_camera_frame(torch.tensor([[[float("nan")]]]), "wrist")

    def test_replay_manager_does_not_restore_unless_explicit(self):
        transition = self._transition()
        with tempfile.TemporaryDirectory() as root:
            journal = Path(root) / "replay.pkl"
            with open(journal, "wb") as stream:
                pickle.dump(transition.to_numpy(), stream)
            buffer = self.module.ReplayBuffer(8)
            self.module.ReplayManager(buffer, str(journal))
            self.assertEqual(len(buffer), 0)

    def test_replay_manager_restores_entries_when_explicit(self):
        # Regression: restore must surface COLLECTION_PHASE_* resolution bugs
        # (constants live in rlt_core.__all__-excluded namespace) instead of
        # silently restoring 0 entries and tripping the resume count check.
        transition = self._transition()
        with tempfile.TemporaryDirectory() as root:
            journal = Path(root) / "replay.pkl"
            with open(journal, "wb") as stream:
                pickle.dump(transition.to_numpy(), stream)
            buffer = self.module.ReplayBuffer(8)
            manager = self.module.ReplayManager(buffer, str(journal), restore=True)
            self.assertEqual(len(buffer), 1)
            self.assertEqual(manager.stats()["adds_total"], 1)

    def test_replay_manager_restore_empty_journal_is_benign(self):
        with tempfile.TemporaryDirectory() as root:
            journal = Path(root) / "replay.pkl"
            journal.write_bytes(b"")
            buffer = self.module.ReplayBuffer(8)
            manager = self.module.ReplayManager(buffer, str(journal), restore=True)
            self.assertEqual(len(buffer), 0)
            self.assertEqual(manager.stats()["adds_total"], 0)

    def test_replay_manager_restore_truncated_tail_keeps_prior_entries(self):
        transition = self._transition()
        with tempfile.TemporaryDirectory() as root:
            journal = Path(root) / "replay.pkl"
            with open(journal, "wb") as stream:
                pickle.dump(transition.to_numpy(), stream)
                pickle.dump(transition.to_numpy(), stream)
            data = journal.read_bytes()
            journal.write_bytes(data[:len(data) - 17])  # cut inside the 2nd record
            buffer = self.module.ReplayBuffer(8)
            manager = self.module.ReplayManager(buffer, str(journal), restore=True)
            self.assertEqual(len(buffer), 1)
            self.assertEqual(manager.stats()["adds_total"], 1)

    def test_replay_manager_restore_legacy_schema_fails_loudly(self):
        # A first-record failure (legacy schema) must propagate, not be
        # swallowed into a misleading "restored count" warning.
        with tempfile.TemporaryDirectory() as root:
            journal = Path(root) / "replay.pkl"
            with open(journal, "wb") as stream:
                pickle.dump({"legacy": True}, stream)
            buffer = self.module.ReplayBuffer(8)
            with self.assertRaises(ValueError):
                self.module.ReplayManager(buffer, str(journal), restore=True)
            self.assertEqual(len(buffer), 0)

    def test_replay_add_is_disk_first(self):
        events = []
        transition = self._transition()
        with tempfile.TemporaryDirectory() as root:
            journal = Path(root) / "replay.pkl"
            buffer = self.module.ReplayBuffer(8)
            original_add = buffer.add

            def checked_add(value):
                events.append(journal.exists() and journal.stat().st_size > 0)
                return original_add(value)

            buffer.add = checked_add
            manager = self.module.ReplayManager(buffer, str(journal))
            manager.add_transition(transition)
            self.assertEqual(events, [True])

    def test_execution_window_materializes_partial_valid_mask_and_actual_human_actions(self):
        boundary = self._boundary(step=3)
        window = self.module.ExecutionWindow(boundary)
        window.append(torch.tensor([0.5, 0.6]), int(self.module.TransitionSource.HUMAN), True)
        window.append(torch.tensor([0.7, 0.8]), int(self.module.TransitionSource.HUMAN), True)
        transition = self.module.build_transition(
            window, self._boundary(step=4), horizon=3, action_dim=2,
        )
        self.assertEqual(transition.executed_steps, 2)
        self.assertEqual(transition.valid_mask.tolist(), [True, True, False])
        self.assertFalse(transition.done)
        self.assertEqual(transition.rewards.tolist(), [0.0, 0.0, 0.0])
        torch.testing.assert_close(
            torch.from_numpy(transition.action_chunk[:2]),
            torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
        )
        torch.testing.assert_close(
            torch.from_numpy(transition.ref_chunk[:2]),
            torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
        )

    def test_reward_and_done_only_apply_at_real_terminal(self):
        window = self.module.ExecutionWindow(self._boundary(step=1))
        window.append(torch.tensor([0.1, 0.2]), int(self.module.TransitionSource.BASE))
        transition = self.module.build_transition(
            window, self._boundary(step=2), horizon=3, action_dim=2,
            reward=1.0, terminal=True,
        )
        self.assertTrue(transition.done)
        self.assertEqual(transition.rewards.tolist(), [1.0, 0.0, 0.0])

    def test_takeover_state_latches_until_successful_handoff(self):
        state = self.module.TakeoverState()
        state.latch(True)
        state.handoff_finished(False)
        self.assertTrue(state.pending)
        state.latch(False)
        self.assertTrue(state.pending)
        state.handoff_finished(True)
        self.assertFalse(state.pending)

    def test_takeover_state_deduplicates_sequences_across_human_chunks(self):
        state = self.module.TakeoverState()
        self.assertTrue(state.accept_leader_sequence(7))
        self.assertFalse(state.accept_leader_sequence(7))
        self.assertFalse(state.accept_leader_sequence(6))
        self.assertTrue(state.accept_leader_sequence(8))

    def test_deferred_windows_keep_partial_policy_and_separate_human_chunks(self):
        policy_start = self._boundary(step=1)
        takeover = self.module.DeferredBoundarySnapshot({}, torch.zeros(2), 2)
        human_end = self.module.DeferredBoundarySnapshot({}, torch.ones(2), 3)
        calls = []

        def materialize(snapshot):
            calls.append(snapshot.step_id)
            return self._boundary(step=snapshot.step_id)

        deferred = [
            self.module.DeferredExecutionWindow(
                policy_start, takeover,
                [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])],
                [int(self.module.TransitionSource.BASE)] * 2,
                [True, True],
            ),
            self.module.DeferredExecutionWindow(
                takeover, human_end,
                [torch.tensor([0.5, 0.6])],
                [int(self.module.TransitionSource.HUMAN)],
                [True],
            ),
        ]
        transitions = self.module.materialize_deferred_windows(
            deferred, materialize, horizon=3, action_dim=2,
        )
        self.assertEqual(calls, [2, 3])
        self.assertEqual([t.executed_steps for t in transitions], [2, 1])
        self.assertEqual(transitions[0].valid_mask.tolist(), [True, True, False])
        self.assertEqual(transitions[1].valid_mask.tolist(), [True, False, False])
        self.assertEqual(transitions[0].source_chunk.tolist(), [0, 0, 0])
        self.assertEqual(transitions[1].source_chunk.tolist(), [2, 0, 0])
        self.assertEqual(transitions[1].step_id, 2)

    def test_deferred_transitions_are_restored_to_temporal_order(self):
        later = self._transition()
        later.step_id = 8
        earlier = self._transition()
        earlier.step_id = 3
        ordered = self.module.order_episode_transitions([later, earlier])
        self.assertEqual([item.step_id for item in ordered], [3, 8])

    def test_clone_observation_detaches_deferred_camera_payload(self):
        frame = torch.ones(2, 2, 3)
        copied = self.module.clone_observation({"wrist": frame})
        frame.zero_()
        self.assertTrue(torch.all(copied["wrist"] == 1))

    def test_leader_sample_requires_fresh_new_sequence(self):
        manager = self.module.TeleopManager(None)
        action = {f"{joint}.pos": 0.0 for joint in manager.JOINT_NAMES}
        sample = self.module.LeaderSample(4, 10.0, action, torch.zeros(6))
        manager._latest_sample = sample
        with mock.patch.object(self.module.time, "monotonic", return_value=10.1):
            self.assertIs(manager.fresh_sample(0.25, after_sequence=3), sample)
            self.assertIsNone(manager.fresh_sample(0.25, after_sequence=4))
        with mock.patch.object(self.module.time, "monotonic", return_value=11.0):
            self.assertIsNone(manager.fresh_sample(0.25))

    def test_leader_waits_for_post_handoff_sequence(self):
        manager = self.module.TeleopManager(None, poll_hz=100.0)
        manager._running = True
        action = {f"{joint}.pos": 0.0 for joint in manager.JOINT_NAMES}
        manager._latest_sample = self.module.LeaderSample(4, 10.0, action, torch.zeros(6))
        with mock.patch.object(self.module.time, "monotonic", side_effect=[10.1, 10.1, 11.2]):
            with mock.patch.object(self.module.time, "sleep"):
                self.assertIsNone(manager.wait_for_fresh_sample(0.25, after_sequence=4, timeout_s=1.0))

    def test_frozen_parameters_preserves_action_gradient_without_critic_grads(self):
        critic = torch.nn.Linear(2, 1)
        action = torch.tensor([[1.0, 2.0]], requires_grad=True)
        with self.module.frozen_parameters(critic):
            critic(action).sum().backward()
        self.assertIsNotNone(action.grad)
        self.assertTrue(all(parameter.grad is None for parameter in critic.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in critic.parameters()))

    def test_checkpoint_builder_uses_paper_full_output_contract(self):
        checkpoint = self.module.build_stage2_checkpoint(total_updates=2)
        self.assertTrue(checkpoint["resume_supported"])
        self.assertEqual(checkpoint["schema_version"], 3)
        self.assertEqual(checkpoint["actor_contract"], "paper_full_output_v1")
        self.assertEqual(checkpoint["rl_chunk_length"], 10)

    def test_hil_pause_resolution_distinguishes_terminal_and_resume(self):
        console = self.module.ConsoleInputManager(io.StringIO())
        for raw, expected in (("y", "success"), ("n", "failure"), ("r", "resume"), ("q", "quit")):
            console._responses.append(raw)
            self.assertEqual(console.prompt_hil_resolution(), expected)

    def test_json_config_loads_structured_fields_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "stage2.json"
            config_path.write_text(json.dumps({
                "pi05_path": "from-config",
                "rlt_checkpoint": "stage1.pt",
                "camera_names": ["top", "wrist"],
                "cameras": {"top": {"type": "opencv", "index_or_path": 1}},
                "camera_map": {"top": "observation.images.top"},
                "dry_run": True,
            }), encoding="utf-8")
            argv = [
                "train_rlt_stage2_pi05.py", "--config", str(config_path),
                "--pi05_path", "from-cli",
            ]
            with mock.patch.object(self.module.sys, "argv", argv):
                args = self.module.parse_args()
            self.assertEqual(args.pi05_path, "from-cli")
            self.assertEqual(args.camera_names, ["top", "wrist"])
            self.assertEqual(json.loads(args.cameras)["top"]["index_or_path"], 1)
            self.assertEqual(json.loads(args.camera_map)["top"], "observation.images.top")
            self.assertTrue(args.dry_run)

    def test_json_config_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "bad.json"
            config_path.write_text(json.dumps({"not_a_real_option": 1}), encoding="utf-8")
            argv = ["train_rlt_stage2_pi05.py", "--config", str(config_path)]
            with mock.patch.object(self.module.sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    self.module.parse_args()

    def test_build_stride2_transitions_emits_overlapping_windows(self):
        module = self.module
        horizon, action_dim, stride = 4, 2, 2
        records = [
            module.StepRecord(
                action=torch.tensor([float(i), float(i + 1)]),
                source=int(module.TransitionSource.BASE),
                intervention=False,
                step_id=100 + i,
            )
            for i in range(8)
        ]
        snapshots = {}
        for step in (0, 2, 4, 6, 8):
            snapshots[step] = module.DeferredBoundarySnapshot(
                observation={},
                state=torch.zeros(6),
                step_id=100 + step,
            )
        terminal = module.DeferredBoundarySnapshot(
            observation={},
            state=torch.zeros(6),
            step_id=100 + 8,
        )

        def materialize(snapshot):
            return module.BoundaryContext(
                z_rl=torch.zeros(1, 2),
                proprio=torch.zeros(6),
                ref_chunk=torch.zeros(1, horizon, action_dim),
                collection_phase="online",
                episode_id=1,
                step_id=snapshot.step_id,
            )

        transitions = module.build_stride2_transitions(
            records,
            snapshots,
            terminal,
            horizon=horizon,
            action_dim=action_dim,
            stride=stride,
            materialize_snapshot=materialize,
        )
        self.assertEqual([t.step_id for t in transitions], [100, 102, 104])
        self.assertEqual([t.executed_steps for t in transitions], [4, 4, 4])

    def _boundary(self, step):
        return self.module.BoundaryContext(
            z_rl=torch.tensor([[1.0, 2.0]]),
            proprio=torch.tensor([0.0, 0.0]),
            ref_chunk=torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]),
            collection_phase="online",
            episode_id=1,
            step_id=step,
        )

    def _transition(self):
        window = self.module.ExecutionWindow(self._boundary(step=1))
        window.append(torch.tensor([0.1, 0.2]), int(self.module.TransitionSource.BASE))
        return self.module.build_transition(
            window, self._boundary(step=2), horizon=3, action_dim=2,
        )


class Stride2ShortEpisodeTest(unittest.TestCase):
    """Regression: a critical phase shorter than one horizon must not drop data."""

    def _build(self, module, n_records, horizon, action_dim, stride):
        records = [
            module.StepRecord(
                action=torch.randn(action_dim),
                source=int(module.TransitionSource.BASE),
                intervention=False,
                step_id=100 + i,
            )
            for i in range(n_records)
        ]
        snapshots = {}
        for step in range(0, n_records + 1, 2):
            snapshots[step] = module.DeferredBoundarySnapshot(
                observation={}, state=torch.zeros(6), step_id=100 + step,
            )
        terminal = module.DeferredBoundarySnapshot(
            observation={}, state=torch.zeros(6), step_id=100 + n_records,
        )

        def materialize(snapshot):
            return module.BoundaryContext(
                z_rl=torch.zeros(1, 2),
                proprio=torch.zeros(6),
                ref_chunk=torch.zeros(1, horizon, action_dim),
                collection_phase="online",
                episode_id=1,
                step_id=snapshot.step_id,
            )

        return module.build_stride2_transitions(
            records, snapshots, terminal,
            horizon=horizon, action_dim=action_dim, stride=stride,
            materialize_snapshot=materialize,
        )

    def test_short_episode_keeps_every_stride_start(self):
        module = _load_stage2_module()
        transitions = self._build(module, n_records=9, horizon=10, action_dim=6, stride=2)
        self.assertEqual([t.step_id - 100 for t in transitions], [0, 2, 4, 6, 8])
        self.assertEqual([t.executed_steps for t in transitions], [9, 7, 5, 3, 1])
        self.assertEqual([t.done for t in transitions], [True, True, True, True, True])

    def test_exact_horizon_episode_emits_one_complete_window(self):
        module = _load_stage2_module()
        transitions = self._build(module, n_records=10, horizon=10, action_dim=6, stride=2)
        self.assertEqual([t.step_id - 100 for t in transitions], [0])
        self.assertEqual([t.executed_steps for t in transitions], [10])
        self.assertTrue(transitions[0].done)

    def test_longer_episode_keeps_complete_and_one_terminal_tail(self):
        module = _load_stage2_module()
        transitions = self._build(module, n_records=13, horizon=10, action_dim=6, stride=2)
        self.assertEqual([t.step_id - 100 for t in transitions], [0, 2, 12])
        self.assertEqual([t.executed_steps for t in transitions], [10, 10, 1])
        self.assertEqual([t.done for t in transitions], [False, False, True])


class Stride2ReferenceAlignmentTest(unittest.TestCase):
    """Stride windows that cross a re-plan boundary must use each row's own
    VLA reference (the chunk that covered that step), not the stale
    window-start chunk reference (paper Eq. (5): BC target shares the same
    time steps as the executed chunk)."""

    def _build(self, module, n_records, horizon, action_dim, stride, human_index=None):
        records = []
        for i in range(n_records):
            is_human = human_index is not None and i in human_index
            records.append(module.StepRecord(
                action=torch.tensor([float(i), float(i + 1)]),
                source=(
                    int(module.TransitionSource.HUMAN) if is_human
                    else int(module.TransitionSource.BASE)
                ),
                intervention=is_human,
                step_id=100 + i,
                ref_row=torch.tensor([float(i), float(i + 100)]),
            ))
        snapshots = {}
        for step in range(0, n_records + 1, stride):
            snapshots[step] = module.DeferredBoundarySnapshot(
                observation={}, state=torch.zeros(6), step_id=100 + step,
            )
        terminal = module.DeferredBoundarySnapshot(
            observation={}, state=torch.zeros(6), step_id=100 + n_records,
        )

        def materialize(snapshot):
            # Deliberately stale: the window-start boundary would give zeros.
            return module.BoundaryContext(
                z_rl=torch.zeros(1, 2),
                proprio=torch.zeros(6),
                ref_chunk=torch.zeros(1, horizon, action_dim),
                collection_phase="online",
                episode_id=1,
                step_id=snapshot.step_id,
            )

        return module.build_stride2_transitions(
            records, snapshots, terminal,
            horizon=horizon, action_dim=action_dim, stride=stride,
            materialize_snapshot=materialize,
        )

    def test_every_row_uses_its_own_step_reference(self):
        module = _load_stage2_module()
        # 12 steps, horizon 10, stride 2 -> windows [0:10] and [2:12]; the
        # latter crosses the re-plan boundary at step 10 (rows 8..9).
        transitions = self._build(module, n_records=12, horizon=10, action_dim=2, stride=2)
        self.assertEqual([t.step_id - 100 for t in transitions], [0, 2])
        for transition, start in ((transitions[0], 0), (transitions[1], 2)):
            for row in range(10):
                expected = torch.tensor([float(start + row), float(start + row + 100)])
                self.assertTrue(
                    torch.allclose(torch.as_tensor(transition.ref_chunk[row]), expected),
                    f"window start {start} row {row} reference mismatch",
                )

    def test_human_rows_keep_executed_action_replacement(self):
        module = _load_stage2_module()
        # Human step at index 2: its reference must stay the executed action
        # (Algorithm 1 intervention replacement), not the per-step ref_row.
        transitions = self._build(
            module, n_records=6, horizon=4, action_dim=2, stride=2, human_index={2},
        )
        window = transitions[0]  # [0:4]
        self.assertEqual(window.source_chunk[2], int(module.TransitionSource.HUMAN))
        torch.testing.assert_close(
            torch.as_tensor(window.ref_chunk[2]),
            torch.as_tensor(window.action_chunk[2]),
        )
        # Non-human rows still use their own step references.
        for row in (0, 1, 3):
            expected = torch.tensor([float(row), float(row + 100)])
            self.assertTrue(torch.allclose(torch.as_tensor(window.ref_chunk[row]), expected))


class ActorModeGatingTest(unittest.TestCase):
    """'a' only enables the RL actor while the critical phase ('c') is active,
    and actor mode is sticky: it is never reset automatically, only by another
    explicit 'a' press."""

    def _feed_key(self, module, key: str, *, critical: bool, actor: bool, expect: bool) -> bool:
        module.CRITICAL_PHASE_ACTIVE = critical
        module.ACTOR_ENABLED = actor
        stream = io.StringIO(key + "\n")
        console = module.ConsoleInputManager(stream)
        console.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and module.ACTOR_ENABLED != expect:
            time.sleep(0.01)
        console.stop()
        return module.ACTOR_ENABLED

    def test_a_does_not_enable_actor_outside_critical_phase(self):
        module = _load_stage2_module()
        self.assertFalse(self._feed_key(module, "a", critical=False, actor=False, expect=False))

    def test_a_enables_actor_inside_critical_phase(self):
        module = _load_stage2_module()
        self.assertTrue(self._feed_key(module, "a", critical=True, actor=False, expect=True))

    def test_second_a_disables_actor_even_inside_critical_phase(self):
        module = _load_stage2_module()
        self.assertFalse(self._feed_key(module, "a", critical=True, actor=True, expect=False))

    def test_actor_mode_is_sticky_across_critical_phase_toggle(self):
        # Turning the critical phase off ('c') must not reset an enabled actor;
        # the actor only turns off on an explicit second 'a'.
        module = _load_stage2_module()
        self.assertTrue(self._feed_key(module, "a", critical=True, actor=False, expect=True))
        module.CRITICAL_PHASE_ACTIVE = False
        self.assertTrue(module.ACTOR_ENABLED)
        self.assertFalse(self._feed_key(module, "a", critical=False, actor=True, expect=False))


class ManualPositioningContractTest(unittest.TestCase):
    """'p' latches a pre-episode positioning request that is strictly separate
    from HIL: it must not touch HIL_TOGGLE_REQUESTED, ACTOR_ENABLED or the
    critical phase, and --manual_positioning must be configurable."""

    def _feed_key(self, module, key: str) -> bool:
        module.POSITION_TOGGLE_REQUESTED = False
        stream = io.StringIO(key + "\n")
        console = module.ConsoleInputManager(stream)
        console.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not module.POSITION_TOGGLE_REQUESTED:
            time.sleep(0.01)
        console.stop()
        return module.POSITION_TOGGLE_REQUESTED

    def test_p_latches_positioning_request(self):
        module = _load_stage2_module()
        self.assertTrue(self._feed_key(module, "p"))

    def test_p_is_independent_of_hil_actor_and_critical_state(self):
        module = _load_stage2_module()
        module.HIL_TOGGLE_REQUESTED = False
        module.ACTOR_ENABLED = False
        module.CRITICAL_PHASE_ACTIVE = False
        self.assertTrue(self._feed_key(module, "p"))
        # Positioning must never bleed into HIL / actor / critical state.
        self.assertFalse(module.HIL_TOGGLE_REQUESTED)
        self.assertFalse(module.ACTOR_ENABLED)
        self.assertFalse(module.CRITICAL_PHASE_ACTIVE)

    def test_manual_positioning_flag_defaults_off_and_is_configurable(self):
        module = _load_stage2_module()
        argv = ["train_rlt_stage2_pi05.py", "--pi05_path", "p", "--rlt_checkpoint", "c", "--dry_run"]
        with mock.patch.object(module.sys, "argv", argv):
            self.assertFalse(module.parse_args().manual_positioning)
        argv = ["train_rlt_stage2_pi05.py", "--pi05_path", "p", "--rlt_checkpoint", "c",
                "--dry_run", "--manual_positioning"]
        with mock.patch.object(module.sys, "argv", argv):
            self.assertTrue(module.parse_args().manual_positioning)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "stage2.json"
            config_path.write_text(json.dumps({
                "pi05_path": "from-config",
                "rlt_checkpoint": "stage1.pt",
                "camera_names": ["top", "wrist"],
                "cameras": {"top": {"type": "opencv", "index_or_path": 1}},
                "camera_map": {"top": "observation.images.top"},
                "dry_run": True,
                "manual_positioning": True,
            }), encoding="utf-8")
            argv = ["train_rlt_stage2_pi05.py", "--config", str(config_path)]
            with mock.patch.object(module.sys, "argv", argv):
                self.assertTrue(module.parse_args().manual_positioning)
            # CLI can disable it even though the config enables it.
            argv = ["train_rlt_stage2_pi05.py", "--config", str(config_path),
                    "--no-manual_positioning"]
            with mock.patch.object(module.sys, "argv", argv):
                self.assertFalse(module.parse_args().manual_positioning)
            # CLI can enable it even though the config disables it.
            config_path.write_text(json.dumps({
                "pi05_path": "from-config",
                "rlt_checkpoint": "stage1.pt",
                "camera_names": ["top", "wrist"],
                "cameras": {"top": {"type": "opencv", "index_or_path": 1}},
                "camera_map": {"top": "observation.images.top"},
                "dry_run": True,
                "manual_positioning": False,
            }), encoding="utf-8")
            argv = ["train_rlt_stage2_pi05.py", "--config", str(config_path),
                    "--manual_positioning"]
            with mock.patch.object(module.sys, "argv", argv):
                self.assertTrue(module.parse_args().manual_positioning)


class RLTNetworkInputContractTest(unittest.TestCase):
    """Regression: MLP input-dimension misuse fails with clear errors, and
    unbatched (chunk, action_dim) tensors are accepted."""

    @staticmethod
    def _config():
        from types import SimpleNamespace
        return SimpleNamespace(rlt_hidden_dim=8, state_dim=2, action_dim=2, n_action_steps_rl=3)

    def test_actor_accepts_flat_2d_and_single_sample_2d_ref_chunk(self):
        from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTChunkActor
        actor = RLTChunkActor(self._config())
        z = torch.randn(2, 8)
        p = torch.randn(2, 2)
        mean, _ = actor(z, p, torch.randn(2, 6))          # (B, chunk*action_dim)
        self.assertEqual(tuple(mean.shape), (2, 3, 2))
        single, _ = actor(z[:1], p[:1], torch.randn(3, 2))  # (chunk, action_dim)
        self.assertEqual(tuple(single.shape), (1, 3, 2))

    def test_actor_rejects_misaligned_ref_chunk_with_clear_error(self):
        from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTChunkActor
        actor = RLTChunkActor(self._config())
        with self.assertRaisesRegex(ValueError, "expected 6"):
            actor(torch.randn(2, 8), torch.randn(2, 2), torch.randn(2, 4, 2))
        with self.assertRaisesRegex(ValueError, "batch 1 != input batch 2"):
            actor(torch.randn(2, 8), torch.randn(2, 2), torch.randn(1, 6))

    def test_critic_rejects_misaligned_action_chunk_with_clear_error(self):
        from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTTwinCritic
        critic = RLTTwinCritic(self._config())
        with self.assertRaisesRegex(ValueError, "expected 6"):
            critic(torch.randn(2, 8), torch.randn(2, 2), torch.randn(2, 4, 2))
