import math
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from gs_agent.algos.config.schema import LearningRateType, PPOArgs
from gs_agent.bases.algo import BaseAlgo
from gs_agent.bases.env_wrapper import BaseEnvWrapper
from gs_agent.bases.policy import Policy
from gs_agent.buffers.config.schema import GAEBufferKey
from gs_agent.buffers.gae_buffer import GAEBuffer
from gs_agent.modules.critics import StateValueFunction
from gs_agent.modules.models import NetworkFactory
from gs_agent.modules.policies import GaussianPolicy

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")
"""Default device for the algorithm."""

_DEQUE_MAXLEN: Final[int] = 100
"""Max length of the deque for storing episode statistics."""

_PER_ENV_DEQUE_MAXLEN: Final[int] = 20
"""Max length of the per-env deque for storing recent episode lengths."""


class PPO(BaseAlgo):
    """
    Proximal Policy Optimization (PPO) algorithm implementation.
    """

    def __init__(
        self, env: BaseEnvWrapper, cfg: PPOArgs, device: torch.device = _DEFAULT_DEVICE
    ) -> None:
        super().__init__(env, cfg, device)
        self._actor_obs_dim = self.env.actor_obs_dim
        self._critic_obs_dim = self.env.critic_obs_dim
        self._action_dim = self.env.action_dim
        self._action_groups: dict[str, slice] = getattr(self.env, "action_groups", {})

        self._num_envs = self.env.num_envs
        self._num_steps = cfg.rollout_length

        self.current_iter = 0
        self._rewbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._lenbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._successbuffer = deque(maxlen=_DEQUE_MAXLEN)  # 1.0 per succeeded episode, else 0.0
        self._per_env_lenbuffer = [deque(maxlen=_PER_ENV_DEQUE_MAXLEN) for _ in range(self._num_envs)]
        self._curr_reward_sum = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.float
        )
        self._curr_ep_len = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float)

        # Adaptive learning rate tracking
        self._current_lr = cfg.lr

        self._build_actor_critic()
        self._build_rollouts()

    def _build_actor_critic(self) -> None:
        policy_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.policy_backbone,
            input_dim=self._actor_obs_dim,
            output_dim=self._action_dim,
            device=self.device,
        )
        print(f"Policy backbone: {policy_backbone}")

        self._actor = GaussianPolicy(
            policy_backbone=policy_backbone,
            action_dim=self._action_dim,
            initial_log_std=self.cfg.initial_log_std,
            state_dependent_std=self.cfg.state_dependent_std,
            log_std_bounds=self.cfg.log_std_bounds,
        ).to(self.device)
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._current_lr)

        critic_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.critic_backbone,
            input_dim=self._critic_obs_dim,
            output_dim=1,
            device=self.device,
        )
        print(f"Critic backbone: {critic_backbone}")
        self._critic = StateValueFunction(critic_backbone).to(self.device)
        value_lr = self.cfg.value_lr if self.cfg.value_lr is not None else self.cfg.lr
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=value_lr)

        print("Device: ", self.device)

    def _build_rollouts(self) -> None:
        self._rollouts = GAEBuffer(
            num_envs=self._num_envs,
            max_steps=self._num_steps,
            actor_obs_size=self._actor_obs_dim,
            critic_obs_size=self._critic_obs_dim,
            action_size=self._action_dim,
            device=self.device,
            gae_gamma=self.cfg.gamma,
            gae_lam=self.cfg.gae_lambda,
        )

    def _update_learning_rate(self, kl_mean: float) -> None:
        """Update learning rate adaptively based on KL divergence."""
        if self.cfg.lr_type == LearningRateType.ADAPTIVE:
            # If KL is too high, reduce learning rate
            if kl_mean > self.cfg.target_kl * 1.5:  # 1.5x threshold
                self._current_lr = max(
                    self._current_lr / self.cfg.lr_adaptive_factor, self.cfg.lr_min
                )
            # If KL is too low, increase learning rate
            elif kl_mean < self.cfg.target_kl * 0.5:  # 0.5x threshold
                self._current_lr = min(
                    self._current_lr * self.cfg.lr_adaptive_factor, self.cfg.lr_max
                )

            # Update only actor optimizer learning rate
            for param_group in self._actor_optimizer.param_groups:
                param_group["lr"] = self._current_lr

    def _collect_rollouts(self, num_steps: int) -> dict[str, Any]:
        """Collect rollouts from the environment."""
        actor_obs, critic_obs = self.env.get_observations()
        termination_buffer = []
        reward_terms_buffer = []

        with torch.inference_mode():
            # collect rollouts
            for _step in range(num_steps):
                action, log_prob, mu, sigma = self._actor.forward_with_dist_params(actor_obs)
                # Step environment
                next_actor_obs, next_critic_obs, env_reward, terminated, truncated, _extra_infos = self.env.step(action)
                # The critic learns values of the scaled reward; episode stats use env_reward
                reward = env_reward * self.cfg.reward_scale

                # add next value to reward of truncated steps
                if "time_outs" in _extra_infos:
                    time_outs = _extra_infos["time_outs"]
                    next_values = self._critic(_extra_infos["observations"]["critic"])
                    reward = reward + next_values * time_outs

                # all tensors are of shape: [num_envs, dim]
                transition = {
                    GAEBufferKey.ACTOR_OBS: actor_obs,
                    GAEBufferKey.CRITIC_OBS: critic_obs,
                    GAEBufferKey.ACTIONS: action,
                    GAEBufferKey.REWARDS: reward,
                    GAEBufferKey.DONES: terminated,
                    GAEBufferKey.VALUES: self._critic(critic_obs),
                    GAEBufferKey.ACTION_LOGPROBS: log_prob,
                    GAEBufferKey.MU: mu,
                    GAEBufferKey.SIGMA: sigma,
                }
                self._rollouts.append(transition)

                # Update episode tracking
                self._curr_reward_sum += env_reward.squeeze(-1)
                self._curr_ep_len += 1

                # Update termination buffer
                termination_buffer.append(_extra_infos["termination"])
                reward_terms_buffer.append(_extra_infos["reward_terms"])

                # Check for episode completions and reset tracking
                done_mask = terminated.unsqueeze(-1) | truncated.unsqueeze(-1)
                new_ids = (done_mask > 0).nonzero(as_tuple=False)
                if len(new_ids) > 0:
                    # Vectorized environment
                    self._rewbuffer.extend(
                        self._curr_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self._lenbuffer.extend(self._curr_ep_len[new_ids][:, 0].cpu().numpy().tolist())
                    succeeded = _extra_infos.get("termination", {}).get("succeeded")
                    if succeeded is not None:
                        self._successbuffer.extend(
                            succeeded[new_ids[:, 0]].float().cpu().numpy().tolist()
                        )
                    # Per-env episode length tracking
                    for idx in new_ids[:, 0].cpu().numpy():
                        self._per_env_lenbuffer[idx].append(self._curr_ep_len[idx].item())
                    # Reset tracking
                    self._curr_reward_sum[new_ids] = 0
                    self._curr_ep_len[new_ids] = 0

                actor_obs, critic_obs = next_actor_obs, next_critic_obs

        with torch.no_grad():
            last_value = self._critic(critic_obs)
            self._rollouts.set_final_value(last_value)

        mean_reward = 0.0
        mean_ep_len = 0.0
        success_rate = 0.0
        if len(self._rewbuffer) > 0:
            mean_reward = statistics.mean(self._rewbuffer)
        if len(self._successbuffer) > 0:
            # Episode-level: fraction of the last episodes that ended by completing the trajectory
            success_rate = statistics.mean(self._successbuffer)
        # Per-env mean episode length: average each env's recent mean
        per_env_means = [statistics.mean(buf) for buf in self._per_env_lenbuffer if len(buf) > 0]
        if per_env_means:
            mean_ep_len = statistics.mean(per_env_means)
        mean_termination = {}
        mean_reward_terms = {}
        if len(termination_buffer) > 0:
            for key in termination_buffer[0].keys():
                terminations = torch.stack([termination[key] for termination in termination_buffer])
                mean_termination[key] = terminations.to(torch.float).mean().item()
        if len(reward_terms_buffer) > 0:
            for key in reward_terms_buffer[0].keys():
                reward_terms = torch.stack(
                    [reward_term[key] for reward_term in reward_terms_buffer]
                )
                mean_reward_terms[key] = reward_terms.mean().item()

        return {
            "mean_reward": mean_reward,
            "mean_ep_len": mean_ep_len,
            "success_rate": success_rate,
            "termination": mean_termination,
            "reward_terms": mean_reward_terms,
        }

    def _train_one_batch(self, mini_batch: dict[GAEBufferKey, torch.Tensor]) -> dict[str, Any]:
        """Train one batch of rollouts."""
        actor_obs = mini_batch[GAEBufferKey.ACTOR_OBS]
        critic_obs = mini_batch[GAEBufferKey.CRITIC_OBS]
        act = mini_batch[GAEBufferKey.ACTIONS]
        old_log_prob = mini_batch[GAEBufferKey.ACTION_LOGPROBS]
        old_mu = mini_batch[GAEBufferKey.MU]
        old_sigma = mini_batch[GAEBufferKey.SIGMA]
        advantage = mini_batch[GAEBufferKey.ADVANTAGES]
        returns = mini_batch[GAEBufferKey.RETURNS]

        new_log_prob = self._actor.evaluate_log_prob(actor_obs, act)
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = -advantage * ratio
        surr2 = -advantage * torch.clamp(
            ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
        )
        policy_loss = torch.max(surr1, surr2).mean()

        # Mean log importance ratio (diagnostic)
        log_ratio = (new_log_prob - old_log_prob).mean()

        # Calculate true KL divergence between old and new distributions
        # Get new distribution parameters
        dist = self._actor.dist_from_obs(actor_obs)
        new_mu = dist.mean
        new_sigma = dist.stddev

        # KL divergence between two Gaussian distributions
        # KL(P||Q) = 0.5 * [log(det(Σ_Q)/det(Σ_P)) - d + tr(Σ_Q^{-1} * Σ_P) + (μ_Q - μ_P)^T * Σ_Q^{-1} * (μ_Q - μ_P)]
        # For diagonal Gaussians: KL = 0.5 * [sum(log(σ_Q^2/σ_P^2)) - d + sum(σ_P^2/σ_Q^2) + sum((μ_Q - μ_P)^2/σ_Q^2)]
        kl_div = 0.5 * (
            torch.log(new_sigma**2 / (old_sigma**2 + 1e-8)).sum(
                dim=-1, keepdim=True
            )  # log ratio of variances
            - new_mu.shape[-1]  # dimension
            + (old_sigma**2 / (new_sigma**2 + 1e-8)).sum(dim=-1, keepdim=True)  # trace term
            + ((new_mu - old_mu) ** 2 / (new_sigma**2 + 1e-8)).sum(
                dim=-1, keepdim=True
            )  # mean difference term
        )
        kl_mean = kl_div.mean()

        # Calculate value loss
        values = self._critic(critic_obs)
        value_loss = (returns - values).pow(2).mean()
        # Fraction of the return variance the critic explains (1 = perfect, <= 0 = no better
        # than predicting the mean). Uses the values predicted during the rollout (the ones GAE
        # used), so later epochs fitting this same minibatch don't inflate it.
        rollout_values = mini_batch[GAEBufferKey.VALUES]
        explained_variance = 1.0 - (returns - rollout_values).var() / (returns.var() + 1e-8)

        # Calculate entropy loss
        entropy = self._actor.entropy_on(actor_obs)
        entropy_loss = entropy.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.cfg.value_loss_coef * value_loss
            - self.cfg.entropy_coef * entropy_loss
        )

        # Optimization step
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._actor.parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self._critic.parameters(), self.cfg.max_grad_norm)
        self._critic_optimizer.step()
        # During critic warmup, skip the actor step so only the critic updates.
        if self.current_iter >= self.cfg.critic_warmup_iters:
            self._actor_optimizer.step()

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "explained_variance": explained_variance.item(),
            # In reward_scale units, like the critic
            "value_mean": rollout_values.mean().item(),
            "value_std": rollout_values.std().item(),
            "return_mean": returns.mean().item(),
            "entropy_loss": entropy_loss.item(),
            "log_ratio": log_ratio.item(),
            "kl_mean": kl_mean.item(),
        }
        # Std and entropy stats from this minibatch's sigma (B, action_dim), averaged over the
        # batch per dim; for a state-independent std every row is the same. Per-dim Gaussian
        # entropy is 0.5 * ln(2*pi*e) + log_std. std_state_spread is the std of sigma across
        # states (0 for a state-independent std).
        with torch.no_grad():
            sigma = new_sigma.detach()
            dim_std = sigma.mean(dim=0)
            dim_entropy = 0.5 * math.log(2 * math.pi * math.e) + sigma.log().mean(dim=0)
            dim_spread = sigma.std(dim=0) if sigma.shape[0] > 1 else torch.zeros_like(dim_std)
            metrics["mean_std"] = dim_std.mean().item()
            metrics["max_std"] = dim_std.max().item()
            metrics["std_state_spread"] = dim_spread.mean().item()
            for name, idx in self._action_groups.items():
                metrics[f"entropy_{name}_per_dim"] = dim_entropy[idx].mean().item()
                metrics[f"mean_std_{name}"] = dim_std[idx].mean().item()
                metrics[f"max_std_{name}"] = dim_std[idx].max().item()
                metrics[f"std_state_spread_{name}"] = dim_spread[idx].mean().item()
        return metrics

    def train_one_iteration(self) -> dict[str, Any]:
        """Train one iteration."""
        # collect rollouts
        t0 = time.time()
        rollout_infos = self._collect_rollouts(num_steps=self._num_steps)
        t1 = time.time()
        rollouts_time = t1 - t0
        fps = (self._num_steps * self._num_envs / rollouts_time) if rollouts_time > 0 else 0

        train_metrics_list: list[dict[str, Any]] = []
        for mini_batch in self._rollouts.minibatch_gen(
            num_mini_batches=self.cfg.num_mini_batches,
            num_epochs=self.cfg.num_epochs,
        ):
            metrics = self._train_one_batch(mini_batch)
            train_metrics_list.append(metrics)

        t2 = time.time()
        train_time = t2 - t1

        # Update learning rate adaptively based on KL divergence
        avg_kl_mean = statistics.mean([metrics["kl_mean"] for metrics in train_metrics_list])
        self._update_learning_rate(avg_kl_mean)

        self._rollouts.reset()

        iteration_infos = {
            "rollout": {
                "mean_reward": rollout_infos["mean_reward"],
                "mean_length": rollout_infos["mean_ep_len"],
                "success_rate": rollout_infos["success_rate"],
            },
            "train": {
                "policy_loss": statistics.mean(
                    [metrics["policy_loss"] for metrics in train_metrics_list]
                ),
                "value_loss": statistics.mean(
                    [metrics["value_loss"] for metrics in train_metrics_list]
                ),
                **{
                    key: statistics.mean([metrics[key] for metrics in train_metrics_list])
                    for key in ("explained_variance", "value_mean", "value_std", "return_mean")
                },
                "entropy_loss": statistics.mean(
                    [metrics["entropy_loss"] for metrics in train_metrics_list]
                ),
                "log_ratio": statistics.mean(
                    [metrics["log_ratio"] for metrics in train_metrics_list]
                ),
                "kl_mean": statistics.mean([metrics["kl_mean"] for metrics in train_metrics_list]),
                "mean_std": statistics.mean(
                    [metrics["mean_std"] for metrics in train_metrics_list]
                ),
                "max_std": statistics.mean([metrics["max_std"] for metrics in train_metrics_list]),
                "std_state_spread": statistics.mean(
                    [metrics["std_state_spread"] for metrics in train_metrics_list]
                ),
                **{
                    key: statistics.mean([metrics[key] for metrics in train_metrics_list])
                    for name in self._action_groups
                    for key in (
                        f"entropy_{name}_per_dim",
                        f"mean_std_{name}",
                        f"max_std_{name}",
                        f"std_state_spread_{name}",
                    )
                },
                "learning_rate": self._current_lr,
            },
            "speed": {
                "rollout_time": rollouts_time,
                "rollout_fps": fps,
                "train_time": train_time,
                "rollout_step": self._num_steps * self._num_envs,
            },
            "termination": rollout_infos["termination"],
            "reward_terms": rollout_infos["reward_terms"],
        }

        # Increment iteration counter
        self.current_iter += 1
        self.env.env.training_step = self.current_iter

        return iteration_infos

    def save(self, path: Path) -> None:
        """Save the algorithm to a file."""
        saved_dict = {
            "model_state_dict": self._actor.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "actor_optimizer_state_dict": self._actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self._critic_optimizer.state_dict(),
            "iter": self.current_iter,
            "current_lr": self._current_lr,
        }
        torch.save(saved_dict, path)

    def load(self, path: Path, load_optimizer: bool = True) -> None:
        """Load the algorithm from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        ckpt_state_dependent = "log_std_head.weight" in checkpoint["model_state_dict"]
        if ckpt_state_dependent != self.cfg.state_dependent_std:
            # strict=False below would otherwise silently drop the std and keep its init value
            raise ValueError(
                f"Checkpoint {path} has state_dependent_std={ckpt_state_dependent}, but the PPO "
                f"config has {self.cfg.state_dependent_std}; pass "
                f"--algo.state_dependent_std={ckpt_state_dependent}."
            )
        missing_actor = self._actor.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing_actor.missing_keys:
            print(f"Actor: missing keys (using defaults): {missing_actor.missing_keys}")

        # Load critic if saved (backward compatibility)
        if "critic_state_dict" in checkpoint:
            missing_critic = self._critic.load_state_dict(checkpoint["critic_state_dict"], strict=False)
            if missing_critic.missing_keys:
                print(f"Critic: missing keys (using defaults): {missing_critic.missing_keys}")

        if load_optimizer:
            self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        self.current_iter = checkpoint["iter"]

        # Restore learning rate if saved (backward compatibility)
        if "current_lr" in checkpoint:
            self._current_lr = checkpoint["current_lr"]
            print(f"Restored learning rate: {self._current_lr}")

    def eval_mode(self) -> None:
        """Set the algorithm to eval mode."""
        self._actor.eval()

    def get_inference_policy(self, device: torch.device | None = None) -> Policy:
        """Get the inference policy for evaluation."""
        self.eval_mode()
        if device is not None:
            self._actor.to(device)
        policy = self._actor
        return policy

