"""Three-stage trainer (paper §III-D).

Stage 1: PPO with teacher z_t                  -> logs/stage1/
Stage 2: freeze policy, train BEV student      -> logs/stage2/
         loss = 0.6*CE(class) + L1(h) + L1(d)
Stage 3: unfreeze both, joint fine-tune        -> logs/stage3/
         loss = L_PPO + 1.0 * L_terrain;  LR x 0.3

Wraps `rsl_rl.runners.OnPolicyRunner` and adds the perception side-loss in
stages 2/3.
"""

from __future__ import annotations

import os
import time
from itertools import chain
from typing import TYPE_CHECKING

import torch
import torch.optim as optim

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan

if TYPE_CHECKING:
    from unitree_dsc_lab.tasks.stair_climb.perception.encoder import BEVStudentEncoder


class ThreeStagePPORunner(OnPolicyRunner):
    """Stage-aware wrapper around rsl_rl's OnPolicyRunner.

    Construction mirrors ``OnPolicyRunner`` but additionally accepts:

    Args:
        env: Vectorised environment (RslRlVecEnvWrapper).
        train_cfg: Runner config dict (BasePPORunnerCfg.to_dict()).
        encoder: Instantiated :class:`BEVStudentEncoder` (not yet on device).
        log_dir: TensorBoard / checkpoint root directory.
        device: Torch device string, e.g. ``"cuda:0"``.
        alpha_terrain: Weight for perception side-loss (paper α = 1).
        encoder_lr: Adam LR for the encoder in stages 2/3.
    """

    def __init__(
        self,
        env,
        train_cfg: dict,
        encoder: "BEVStudentEncoder",
        log_dir: str | None = None,
        device: str = "cpu",
        *,
        alpha_terrain: float = 1.0,
        encoder_lr: float = 1e-3,
    ) -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.encoder = encoder.to(device)
        self.alpha_terrain = alpha_terrain
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=encoder_lr)

    # -----------------------------------------------------------------------
    # Stage 1 — PPO with ground-truth teacher z_t
    # -----------------------------------------------------------------------

    def learn_stage1(self, num_iterations: int, init_at_random_ep_len: bool = True) -> None:
        """Pure PPO pre-training using the privileged teacher token as z_t.

        Target: success_rate > 0.85 on training stair heights (paper Table II).
        """
        self.learn(num_iterations, init_at_random_ep_len=init_at_random_ep_len)

    # -----------------------------------------------------------------------
    # Stage 2 — Supervised BEV encoder training (policy frozen)
    # -----------------------------------------------------------------------

    def learn_stage2(
        self,
        num_epochs: int,
        batch_size: int = 256,
        sl_epochs_per_rollout: int = 5,
    ) -> None:
        """Freeze policy; collect (BEV, z_t_gt) rollouts; train encoder.

        Each epoch:
          1. Roll out the frozen actor for ``num_steps_per_env`` steps, collecting
             ``(bev, z_t_gt)`` pairs via ``env.get_bev()`` +
             ``env.privileged_teacher.token()``.
          2. Run ``sl_epochs_per_rollout`` SL passes over the buffer using
             :func:`~unitree_dsc_lab.tasks.stair_climb.perception.encoder.terrain_loss`.

        The env must expose ``get_bev() -> Tensor(num_envs, 6, 60, 60)``.
        A :exc:`RuntimeError` is raised immediately if absent so the
        mis-configuration is caught at startup rather than mid-epoch.

        Targets (paper Table II): MAE(h) < 1 cm, MAE(d) < 1 cm, class_acc > 99 %.
        """
        env_unwrapped = self.env.unwrapped
        if not hasattr(env_unwrapped, "get_bev"):
            raise RuntimeError(
                "Stage 2 requires env.get_bev() -> Tensor(num_envs, 6, 60, 60). "
                "Add a simulated depth sensor and implement get_bev() on the env "
                "or in a RslRlVecEnvWrapper subclass. "
                "See deploy/pointcloud_to_bev.py for the real-world counterpart."
            )

        # Freeze PPO actor + critic
        for p in chain(self.alg.actor.parameters(), self.alg.critic.parameters()):
            p.requires_grad_(False)
        self.alg.eval_mode()
        self.encoder.train()

        obs = self.env.get_observations().to(self.device)

        for epoch in range(num_epochs):
            bev_list: list[torch.Tensor] = []
            z_gt_list: list[torch.Tensor] = []

            # Collect (BEV, z_t_gt) pairs over one rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, _, dones, _ = self.env.step(actions.to(self.env.device))
                    obs = obs.to(self.device)
                    bev_list.append(env_unwrapped.get_bev().cpu())
                    z_gt_list.append(self._teacher_token(env_unwrapped).cpu())
                    # reset recurrent state (no-op for MLP policy)
                    self.alg.actor.reset(dones)
                    self.alg.critic.reset(dones)

            bev_all = torch.cat(bev_list, dim=0).to(self.device)    # (T*E, 6, 60, 60)
            z_gt_all = torch.cat(z_gt_list, dim=0).to(self.device)  # (T*E, 4)

            losses = self._run_sl_update(bev_all, z_gt_all, batch_size, sl_epochs_per_rollout)
            print(
                f"[Stage 2] epoch {epoch + 1:03d}/{num_epochs}  "
                f"total={losses['total']:.4f}  cls={losses['cls']:.4f}  "
                f"h={losses['h']:.4f}  d={losses['d']:.4f}"
            )

        # Unfreeze PPO actor + critic
        for p in chain(self.alg.actor.parameters(), self.alg.critic.parameters()):
            p.requires_grad_(True)

    # -----------------------------------------------------------------------
    # Stage 3 — Joint fine-tuning: L_PPO + alpha * L_terrain
    # -----------------------------------------------------------------------

    def learn_stage3(
        self,
        num_iterations: int,
        lr_scale: float = 0.3,
        init_at_random_ep_len: bool = True,
    ) -> None:
        """Joint PPO + encoder fine-tuning.

        Loss: L_total = L_PPO + alpha * L_terrain  (paper §III-D, α = 1).

        Both the PPO and encoder optimisers are scaled down by ``lr_scale``
        (paper recommends ×0.3).

        If the env does not expose ``get_bev()``, the perception side-loss is
        skipped and Stage 3 reduces to vanilla PPO.

        Rollouts still use teacher z_t for the policy observation to keep the
        actor stable while the encoder catches up. Full end-to-end student z_t
        through-the-policy gradients are a future extension.
        """
        env_unwrapped = self.env.unwrapped
        has_bev = hasattr(env_unwrapped, "get_bev")
        if not has_bev:
            print(
                "[Stage 3] Warning: env.get_bev() not found — "
                "running PPO only (encoder not updated)."
            )

        # Scale learning rates
        for g in self.alg.optimizer.param_groups:
            g["lr"] *= lr_scale
        self.alg.learning_rate *= lr_scale
        for g in self.encoder_optimizer.param_groups:
            g["lr"] *= lr_scale

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.encoder.train()

        if self.is_distributed:
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_iterations

        for it in range(start_it, total_it):
            t_start = time.time()
            bev_list: list[torch.Tensor] = []
            z_gt_list: list[torch.Tensor] = []

            # Rollout (mirrors OnPolicyRunner.learn inner loop)
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)
                    if has_bev:
                        bev_list.append(env_unwrapped.get_bev().cpu())
                        z_gt_list.append(self._teacher_token(env_unwrapped).cpu())

            collect_time = time.time() - t_start
            t_start = time.time()

            self.alg.compute_returns(obs)

            # PPO update
            loss_dict = self.alg.update()

            # Encoder perception side-loss — separate optimiser, no interference with PPO
            if has_bev and bev_list:
                bev_all = torch.cat(bev_list, dim=0).to(self.device)
                z_gt_all = torch.cat(z_gt_list, dim=0).to(self.device)
                perc = self._run_sl_update(
                    bev_all,
                    z_gt_all,
                    batch_size=max(256, bev_all.shape[0] // 4),
                    num_epochs=1,
                )
                loss_dict["terrain"] = perc["total"]
                loss_dict["terrain_cls"] = perc["cls"]
                loss_dict["terrain_h"] = perc["h"]
                loss_dict["terrain_d"] = perc["d"]

            learn_time = time.time() - t_start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    # -----------------------------------------------------------------------
    # Stage 1 evaluation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_success_rate(
        self,
        num_episodes: int = 200,
        min_level_frac: float = 0.8,
    ) -> float:
        """Estimate Stage-1 success rate from terrain-level progress.

        Runs the frozen policy until ``num_episodes`` env episodes complete, then
        returns the fraction that ended at ``terrain_level >= min_level_frac *
        max_level``.  Returns ``nan`` if the terrain has no level tracking.

        Args:
            num_episodes: Minimum number of completed episodes to collect.
            min_level_frac: Fraction of max terrain levels considered a success
                (default 0.8 ≈ reaching the top-2 difficulty rows out of 10).
        """
        env_unwrapped = self.env.unwrapped
        terrain = getattr(getattr(env_unwrapped, "scene", None), "terrain", None)
        if terrain is None or not hasattr(terrain, "terrain_levels"):
            return float("nan")

        max_level = int(terrain.cfg.terrain_generator.num_rows) - 1
        threshold = int(round(max_level * min_level_frac))

        self.alg.eval_mode()
        obs = self.env.get_observations().to(self.device)
        successes = 0
        total = 0

        while total < num_episodes:
            with torch.inference_mode():
                actions = self.alg.act(obs)
                obs, _, dones, _ = self.env.step(actions.to(self.env.device))
                obs = obs.to(self.device)
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                levels = terrain.terrain_levels[done_ids.cpu()]
                successes += int((levels >= threshold).sum().item())
                total += int(done_ids.numel())

        self.alg.train_mode()
        return successes / max(total, 1)

    # -----------------------------------------------------------------------
    # Checkpoint helpers (extend parent to include encoder)
    # -----------------------------------------------------------------------

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save PPO + encoder state to a single checkpoint file."""
        saved = self.alg.save()
        saved["iter"] = self.current_learning_iteration
        saved["infos"] = infos
        saved["encoder_state_dict"] = self.encoder.state_dict()
        saved["encoder_optimizer_state_dict"] = self.encoder_optimizer.state_dict()
        torch.save(saved, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Load PPO + encoder state; returns infos dict."""
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded["iter"]
        if "encoder_state_dict" in loaded:
            self.encoder.load_state_dict(loaded["encoder_state_dict"], strict=strict)
        if "encoder_optimizer_state_dict" in loaded:
            self.encoder_optimizer.load_state_dict(loaded["encoder_optimizer_state_dict"])
        return loaded.get("infos")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _teacher_token(self, env_unwrapped) -> torch.Tensor:
        """Return (num_envs, 4) GT z_t from the privileged teacher."""
        teacher = getattr(env_unwrapped, "privileged_teacher", None)
        if teacher is None:
            raise RuntimeError(
                "env.privileged_teacher not found. "
                "Ensure reset_privileged_teacher is in EventCfg."
            )
        robot_yaw = env_unwrapped.scene["robot"].data.heading_w
        return teacher.token(robot_yaw)

    def _run_sl_update(
        self,
        bev: torch.Tensor,
        z_gt: torch.Tensor,
        batch_size: int,
        num_epochs: int,
    ) -> dict[str, float]:
        """Mini-batch SL update for the encoder; returns mean losses."""
        from unitree_dsc_lab.tasks.stair_climb.perception.encoder import terrain_loss

        N = bev.shape[0]
        totals = {"total": 0.0, "cls": 0.0, "h": 0.0, "d": 0.0}
        n_updates = 0

        self.encoder.train()
        for _ in range(num_epochs):
            perm = torch.randperm(N, device=bev.device)
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                pred = self.encoder(bev[idx])
                losses = terrain_loss(pred, z_gt[idx])
                self.encoder_optimizer.zero_grad()
                losses["total"].backward()
                self.encoder_optimizer.step()
                for k in totals:
                    totals[k] += losses[k].item()
                n_updates += 1

        n = max(n_updates, 1)
        return {k: v / n for k, v in totals.items()}
