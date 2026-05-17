"""Paper §III actor/critic over proprioception ⊕ z_t (Eq. 1, Fig. 2).

The paper's policy is a plain MLP — the same architecture for actor and critic
heads — and is fed the concatenation of proprioceptive observations with the
4-D terrain token. We therefore implement it as a thin subclass of
:class:`rsl_rl.models.MLPModel` so it is a drop-in for rsl_rl's
:class:`OnPolicyRunner` / PPO construction path. Reference the class from the
runner cfg via the qualified name
``"unitree_dsc_lab.tasks.stair_climb.policy.actor_critic:StairClimbActorCritic"``
(``rsl_rl.utils.resolve_callable`` accepts ``"module.path:ClassName"``).

Stage wiring:

* **Stage 1** — the env's ``policy`` obs group ends with the 4 channels of
  ``mdp.terrain_token_privileged`` (teacher GT), so the actor sees
  ``o_prop ⊕ z_t`` exactly as the paper describes.
* **Stage 2** — the policy is frozen; the BEV student
  ([`perception.encoder.BEVStudentEncoder`])
  is supervised against teacher tokens (no change here).
* **Stage 3** — swap the ``terrain_token`` obs term for one backed by the
  student encoder. The actor still consumes ``o_prop ⊕ z_t`` — the gradient
  flow to the encoder is handled at the runner level
  (see ``policy/ppo_runner.py``).

The class is intentionally minimal: no architectural change vs. ``MLPModel``,
just paper-aligned defaults and a ``paper_terrain_token_dim`` class attribute
that documents how the trailing 4 obs channels are interpreted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rsl_rl.models import MLPModel

if TYPE_CHECKING:
    from tensordict import TensorDict


class StairClimbActorCritic(MLPModel):
    """Stock MLP actor/critic specialized for the paper's stair-climbing task.

    Use the same class for both the actor (with a Gaussian distribution cfg)
    and the critic (deterministic, ``output_dim=1``). Two roles are
    distinguished by the ``obs_set`` argument that ``MLPModel.__init__``
    receives from ``PPO.construct_algorithm``.

    Class attributes:
        paper_terrain_token_dim: number of channels in the appended 4-D
            terrain token ``z_t = [class_id, h_step, d_step, theta_yaw]``.
            Matches :func:`mdp.terrain_token_privileged` and
            :meth:`BEVStudentEncoder.predict_token`.
        is_recurrent: always ``False`` (paper §III-A uses a memoryless MLP).
    """

    paper_terrain_token_dim: int = 4
    is_recurrent: bool = False

    def __init__(
        self,
        obs: "TensorDict",
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )


__all__ = ["StairClimbActorCritic"]
