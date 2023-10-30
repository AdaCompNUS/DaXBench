from daxbench.core.envs.registration import env_functions
import jax.numpy as jnp
from average_meter import AverageMeter
from time_report import TimeReport
from dataset import CriticDataset
import yaml
import copy
import time
import os
import sys

import jax
import numpy as np
import optax
import pickle
import imageio
import argparse
from tensorboardX import SummaryWriter
from brax.training import distribution, normalization
from daxbench.algorithms.shac.actor_jax import make_actor_model, make_critic_model
from daxbench.core.envs.shape_rope_env import ShapeRopeEnv

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

my_path = os.path.dirname(os.path.abspath(__file__))


def clip_by_global_norm(updates, max_gradient_norm):
    g_norm = optax.global_norm(updates)
    trigger = g_norm < max_gradient_norm
    updates = jax.tree_util.tree_map(
        lambda t: jnp.where(trigger, t, (t / g_norm) *
                            max_gradient_norm), updates
    )

    return updates


class SHAC:
    def __init__(self, cfg, args):
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr

        # seeds
        batch_size = args.num_envs
        self.batch_size = batch_size
        self.eval_freq = args.eval_freq
        self.grad_norm = args.max_grad_norm
        self.max_epochs = args.max_it

        key = jax.random.PRNGKey(args.seed)
        key, key_models, self.key_env = jax.random.split(key, 3)
        self.key_eval = jax.random.PRNGKey(args.seed + 666)

        # simulation envs
        core_env = env_functions[args.env](
            batch_size=self.batch_size, aux_reward=True, seed=args.seed
        )
        eval_env = env_functions[args.env](
            batch_size=self.batch_size, seed=args.seed + 666
        )
        self.eval_env = eval_env
        self.core_env = core_env
        self.step_fn = self.core_env.step_diff
        self.eval_step = self.eval_env.step_diff
        self.eval_reset = self.eval_env.reset
        self.reset_fn = self.core_env.reset

        args.ep_len = core_env.max_steps
        episode_length = args.ep_len
        self.steps_num = episode_length
        args.logdir = f"logs/shac/{args.env}/{args.env}_ep_len{args.ep_len}_num_envs{args.num_envs}_actor_lr{args.actor_lr}" \
                      f"critic_lr{args.critic_lr}_max_it{args.max_it}_max_grad_norm{args.max_grad_norm}/seed{args.seed}"

        self.logdir = args.logdir
        self.env = args.env
        self.writer = SummaryWriter(args.logdir)

        # Get initial state
        if isinstance(core_env, ShapeRopeEnv):
            self.first_state = self.reset_fn(self.key_env)[1]
            self.auto_reset = self.core_env.auto_reset

        self.eval_first_state = self.eval_env.reset(self.key_eval)

        # init optimizer
        self.save_interval = 20
        self.action_dist = distribution.NormalTanhDistribution(
            event_size=core_env.action_size
        )
        self.actor_model = make_actor_model(
            self.action_dist, core_env.observation_size)
        self.actor_params = self.actor_model.init(key_models)
        self.actor_optimizer = optax.adam(learning_rate=self.actor_lr)
        self.act_opt_state = self.actor_optimizer.init(self.actor_params)

        self.critic_model = make_critic_model(core_env.observation_size)
        self.critic_params = self.critic_model.init(key_models)
        self.critic_optimizer = optax.adam(learning_rate=self.critic_lr)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

        # observation normalizer
        (
            self.normalizer_params,
            self.obs_normalizer_update_fn,
            self.obs_normalizer_apply_fn,
        ) = normalization.create_observation_normalizer(
            core_env.observation_size, True, num_leading_batch_dims=2
        )

        self.loss_grad = None
        self.critic_loss_grad = None

        print("num_envs = ", batch_size)
        print("num_actions = ", core_env.action_size)
        print("num_obs = ", core_env.observation_size)

        self.num_envs = batch_size
        self.num_obs = core_env.observation_size
        self.num_actions = core_env.action_size
        self.max_episode_length = episode_length
        self.gamma = args.gamma

        self.critic_method = cfg["params"]["config"].get(
            "critic_method", "one-step"
        )  # ['one-step', 'td-lambda']
        if self.critic_method == "td-lambda":
            self.lam = cfg["params"]["config"].get("lambda", 0.95)

        self.lr_schedule = cfg["params"]["config"].get("lr_schedule", "linear")

        self.target_critic_alpha = cfg["params"]["config"].get(
            "target_critic_alpha", 0.4
        )
        # self.rew_scale = cfg['params']['config'].get('rew_scale', 1.0)
        self.rew_scale = 1.0

        self.critic_iterations = cfg["params"]["config"].get(
            "critic_iterations", 16)
        self.truncate_grad = cfg["params"]["config"]["truncate_grads"]

        # create actor critic network
        self.actor_name = cfg["params"]["network"].get(
            "actor", "ActorStochasticMLP")
        self.critic_name = cfg["params"]["network"].get("critic", "CriticMLP")
        self.all_params = [self.actor_params, self.critic_params]
        self.target_critic_para = copy.deepcopy(self.critic_params)

        # replay buffer
        self.obs_buf = jnp.zeros(
            (self.steps_num, self.num_envs, self.num_obs), dtype=jnp.float32
        )
        self.rew_buf = jnp.zeros(
            (self.steps_num, self.num_envs), dtype=jnp.float32)
        self.done_mask = jnp.zeros(
            (self.steps_num, self.num_envs), dtype=jnp.float32)
        self.next_values = jnp.zeros(
            (self.steps_num, self.num_envs), dtype=jnp.float32)
        self.target_values = jnp.zeros(
            (self.steps_num, self.num_envs), dtype=jnp.float32
        )
        self.ret = jnp.zeros((self.num_envs), dtype=jnp.float32)

        # for kl divergence computing
        self.old_mus = jnp.zeros(
            (self.steps_num, self.num_envs, self.num_actions), dtype=jnp.float32
        )
        self.old_sigmas = jnp.zeros(
            (self.steps_num, self.num_envs, self.num_actions), dtype=jnp.float32
        )
        self.mus = jnp.zeros(
            (self.steps_num, self.num_envs, self.num_actions), dtype=jnp.float32
        )
        self.sigmas = jnp.zeros(
            (self.steps_num, self.num_envs, self.num_actions), dtype=jnp.float32
        )

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # loss variables
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_loss = jnp.zeros(self.num_envs, dtype=jnp.float32)
        self.episode_discounted_loss = jnp.zeros(
            self.num_envs, dtype=jnp.float32)
        self.episode_gamma = jnp.ones(self.num_envs, dtype=jnp.float32)
        self.episode_length = jnp.zeros(self.num_envs, dtype=int)
        self.best_policy_loss = jnp.inf
        self.actor_loss = jnp.inf
        self.value_loss = jnp.inf

        # average meter
        self.episode_loss_meter = AverageMeter(1, 100)
        self.episode_discounted_loss_meter = AverageMeter(1, 100)
        self.episode_length_meter = AverageMeter(1, 100)

        # timer
        self.time_report = TimeReport()

    def compute_actor_loss_jax(self):
        def do_one_step(carry, i):
            (
                actor_params,
                state_obs,
                normalizer_params,
                key,
                next_values,
                target_critic_para,
                episode_length,
                rew_acc,
                actor_loss,
                gamma,
            ) = carry
            key, key_sample = jax.random.split(key)

            obs = state_obs[0]
            state = state_obs[1]
            normalized_obs = obs

            logits = self.actor_model.apply(actor_params, normalized_obs)
            actions = self.action_dist.sample(logits, key_sample)
            if self.env == 'shape_rope':
                actions = jax.nn.sigmoid(actions)
            next_values = next_values.at[i + 1].set(
                self.critic_model.apply(
                    target_critic_para, normalized_obs).squeeze(-1)
            )

            obs, reward, done, info = self.step_fn(actions, state)
            inv_done = (1.0 - done).astype(jnp.int32)  # scale the reward
            reward = reward * self.rew_scale
            episode_length += 1

            # early termination
            early_mask = (episode_length <
                          self.max_episode_length).astype(jnp.int32)
            early_mask *= done
            next_values = next_values.at[i +
                                         1].set(next_values[i + 1] * early_mask)

            # obs_before_reset
            # TODO, update values for obs_before_reset

            # Shape of the reward is (N,)
            reward = reward.flatten()
            rew_acc = rew_acc.at[i + 1, :].set(rew_acc[i, :] + gamma * reward)
            actor_loss_non_end = (
                    actor_loss
                    + (
                            -rew_acc[i + 1] * done
                            - self.gamma * gamma * done * next_values[i + 1] * done
                    ).sum()
            )
            # else
            actor_loss_end = (
                    actor_loss
                    + (
                            -rew_acc[i + 1, :] - self.gamma *
                            gamma * next_values[i + 1, :]
                    ).sum()
            )
            actor_loss = jnp.where(
                i < self.steps_num - 1, actor_loss_non_end, actor_loss_end
            )
            # compute gamma for next step
            gamma = gamma * self.gamma
            gamma = gamma * inv_done + done
            rew_acc = rew_acc.at[i + 1].set(rew_acc[i + 1] * inv_done)

            # collect data for critic training
            episode_length *= 1 - done.astype(jnp.int32)
            state_obs = (obs, info["state"])
            obs = obs.reshape((obs.shape[0], -1))
            next_carry = (
                actor_params,
                state_obs,
                normalizer_params,
                key,
                next_values,
                target_critic_para,
                episode_length,
                rew_acc,
                actor_loss,
                gamma,
            )

            return next_carry, (obs, reward, done)

        def loss(actor_params, carry):
            carry = (actor_params,) + carry
            next_carry, (obs, rew, done_mask) = jax.lax.scan(
                do_one_step,
                carry,
                (jnp.array(range(self.steps_num))),
                length=self.steps_num,
            )
            (
                actor_params,
                nstate,
                normalizer_params,
                key,
                next_values,
                target_critic_para,
                episode_length,
                rew_acc,
                actor_loss,
                gamma,
            ) = next_carry

            actor_loss /= self.steps_num * self.num_envs

            return actor_loss, (actor_loss, obs, rew, done_mask, next_values, nstate)

        def _minimize(state):
            self.key_env, key_grad = jax.random.split(self.key_env)
            rew_acc = jnp.zeros(
                (self.steps_num + 1, self.num_envs), dtype=jnp.float32)
            gamma = jnp.ones(self.num_envs, dtype=jnp.float32)
            next_values = jnp.zeros(
                (self.steps_num + 1, self.num_envs), dtype=jnp.float32
            )
            actor_loss = jnp.array(0.0, dtype=jnp.float32)
            episode_length = jnp.zeros(self.num_envs, dtype=jnp.int32)

            carry = (
                state,
                self.normalizer_params,
                self.key_env,
                next_values,
                self.target_critic_para,
                episode_length,
                rew_acc,
                actor_loss,
                gamma,
            )

            grad_raw, (
                actor_loss,
                normalized_obs,
                rew,
                done_mask,
                next_values,
                nstate,
            ) = self.loss_grad(self.actor_params, carry)

            # collect data for critic training
            self.sim_state = nstate
            self.actor_loss = np.array(actor_loss)
            self.step_count += self.steps_num * self.num_envs
            self.obs_buf = normalized_obs
            self.rew_buf = rew
            self.done_mask = done_mask
            self.next_values = next_values

            for i in range(len(rew)):
                self.episode_loss -= rew[i]
                self.episode_length += 1
                done_env_ids = done_mask[i].nonzero()[0]
                if len(done_env_ids) > 0:
                    self.episode_loss_meter.update(
                        self.episode_loss[done_env_ids])
                    self.episode_length_meter.update(
                        self.episode_length[done_env_ids])
                    for done_env_id in done_env_ids:
                        self.episode_loss_his.append(
                            self.episode_loss[done_env_id])
                        self.episode_loss = self.episode_loss.at[done_env_id].set(
                            0)
                        self.episode_length_his.append(
                            self.episode_length[done_env_id])
                        self.episode_length = self.episode_length.at[done_env_id].set(
                            0)

            # update actor params
            grad_raw = jax.tree_util.tree_map(
                lambda t: jnp.nan_to_num(t), grad_raw)
            grad = clip_by_global_norm(grad_raw, self.grad_norm)
            params_update, self.act_opt_state = self.actor_optimizer.update(
                grad, self.act_opt_state
            )

            self.actor_params = optax.apply_updates(
                self.actor_params, params_update)

            return optax.global_norm(grad)

        if self.loss_grad is None:
            loss = jax.jit(loss)
            self.loss_grad = jax.grad(loss, has_aux=True)

        grad = _minimize(self.sim_state)
        return grad

    def evaluate_policy(self, step, deterministic=False):

        def do_one_step_eval(carry, i):
            actor_params, state_obs, normalizer_params, key = carry
            key, key_sample = jax.random.split(key)

            obs = state_obs[0]
            state = state_obs[1]
            normalized_obs = obs

            logits = self.actor_model.apply(actor_params, normalized_obs)
            actions = self.action_dist.sample(logits, key_sample)

            # Warning: remove it when run on whip rope
            if self.env == 'shape_rope':
                actions = jax.nn.sigmoid(actions)
            obs, reward, done, info = self.eval_step(actions, state)
            next_carry = actor_params, (obs,
                                        info["state"]), normalizer_params, key

            return next_carry, (obs, reward, done, actions, state)

        def run_eval(actor_params, carry):
            carry = (actor_params,) + carry
            next_carry, (obs, rew, done_mask, action_list, state_list) = jax.lax.scan(
                do_one_step_eval,
                carry,
                (jnp.array(range(self.core_env.max_steps))),
                length=self.core_env.max_steps,
            )

            actor_params, nstate, normalizer_params, key = next_carry

            return obs, rew, done_mask, nstate, action_list, state_list, key

        obs, rew, done_mask, nstate, action_list, state_list, key = run_eval(
            self.actor_params, (self.eval_first_state,
                                self.normalizer_params, self.key_eval)
        )
        # visualize
        rgb_list = []
        for i in range(self.core_env.max_steps):
            state = jax.tree_util.tree_map(lambda x: x[i], state_list)
            obs, reward, done, info = self.eval_env.step_with_render(
                action_list[i], state, visualize=False)
            rgb_list.extend(info["img_list"])
        # save rgb_list into gif file named "fold_cloth_{it}.gif"
        os.makedirs(self.logdir, exist_ok=True)
        imageio.mimsave(
            f"./{self.logdir}/shac_{self.env}_{step}.gif", rgb_list, fps=20)
        return rew

    def compute_target_values(self):
        if self.critic_method == "one-step":
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == "td-lambda":
            Ai = jnp.zeros(self.num_envs, dtype=jnp.float32)
            Bi = jnp.zeros(self.num_envs, dtype=jnp.float32)
            lam = jnp.ones(self.num_envs, dtype=jnp.float32)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * \
                      (1.0 - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (
                        self.lam * self.gamma * Ai
                        + self.gamma * self.next_values[i]
                        + (1.0 - lam) / (1.0 - self.lam) * self.rew_buf[i]
                )
                Bi = (
                        self.gamma
                        * (
                                self.next_values[i] * self.done_mask[i]
                                + Bi * (1.0 - self.done_mask[i])
                        )
                        + self.rew_buf[i]
                )
                self.target_values = self.target_values.at[i].set(
                    (1.0 - self.lam) * Ai + lam * Bi
                )
        else:
            raise NotImplementedError

    def train(self):
        self.start_time = time.time()

        # add timers
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("compute actor loss")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("backward simulation")
        self.time_report.add_timer("prepare critic dataset")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")
        self.time_report.start_timer("algorithm")

        # initializations
        self.episode_loss = jnp.zeros(self.num_envs, dtype=jnp.float32)
        self.episode_discounted_loss = jnp.zeros(
            self.num_envs, dtype=jnp.float32)
        self.episode_length = jnp.zeros(self.num_envs, dtype=int)
        self.episode_gamma = jnp.ones(self.num_envs, dtype=jnp.float32)

        def actor_closure_jax():
            self.time_report.start_timer("compute actor loss")
            grad = self.compute_actor_loss_jax()
            print("grad: ", grad)
            self.time_report.end_timer("compute actor loss")

        def compute_critic_loss_jax(critic_para, batch_sample):
            predicted_values = self.critic_model.apply(
                critic_para, batch_sample["obs"]
            )
            target_values = batch_sample["target_values"]
            critic_loss = ((predicted_values - target_values) ** 2).mean()

            return critic_loss, critic_loss

        if self.critic_loss_grad is None:
            critic_loss_grad = jax.jit(compute_critic_loss_jax)
            self.critic_loss_grad = jax.grad(critic_loss_grad, has_aux=True)
            self.critic_loss_grad = jax.jit(self.critic_loss_grad)

        # main training process
        self.sim_state = self.reset_fn(self.key_env)
        self.key_env, self.key_reset = jax.random.split(self.key_env)

        for epoch in range(self.max_epochs):
            if self.env == "shape_rope":
                self.key_reset = jax.random.split(self.key_reset)
                self.sim_state = self.reset_fn(self.key_reset)

            if epoch % self.eval_freq == 0:
                rew = self.evaluate_policy(epoch)
                self.save(f'{self.logdir}_epoch_{epoch}_self.env')
                print(f"epoch {epoch}, reward {rew.mean()}")
                self.writer.add_scalar("reward", rew.sum(0).mean(), epoch)
                self.writer.add_scalar("last_reward", rew[-1].mean(), epoch)

            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == "linear":
                actor_lr = (1e-5 - self.actor_lr) * float(
                    epoch / self.max_epochs
                ) + self.actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(
                    epoch / self.max_epochs
                ) + self.critic_lr
                lr = actor_lr

                self.critic_optimizer = optax.adam(learning_rate=critic_lr)
                self.actor_optimizer = optax.adam(learning_rate=actor_lr)

            else:
                lr = self.actor_lr

            # train actor
            self.time_report.start_timer("actor training")
            actor_closure_jax()
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            self.compute_target_values()
            dataset = CriticDataset(
                self.batch_size, self.obs_buf, self.target_values, drop_last=False
            )
            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.0
            for j in range(self.critic_iterations):
                total_critic_loss = 0.0
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    grad_raw, training_critic_loss = self.critic_loss_grad(
                        self.critic_params, batch_sample
                    )
                    # update actor params
                    grad_raw = jax.tree_util.tree_map(
                        lambda t: jnp.nan_to_num(t), grad_raw
                    )
                    grad = clip_by_global_norm(grad_raw, self.grad_norm)
                    params_update, self.critic_opt_state = self.critic_optimizer.update(
                        grad, self.critic_opt_state
                    )
                    self.critic_params = optax.apply_updates(
                        self.critic_params, params_update
                    )

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1

                self.value_loss = jnp.array(total_critic_loss / batch_cnt)
                print(
                    "value iter {}/{}, loss = {:7.6f}".format(
                        j + 1, self.critic_iterations, self.value_loss
                    ),
                    end="\r",
                )

            self.time_report.end_timer("critic training")
            self.iter_count += 1

            time_end_epoch = time.time()

            # logging
            time_elapse = time.time() - self.start_time
            self.writer.add_scalar('lr/iter', lr, self.iter_count)
            self.writer.add_scalar(
                'actor_loss/step', self.actor_loss, self.step_count)
            self.writer.add_scalar(
                'actor_loss/iter', self.actor_loss, self.iter_count)
            self.writer.add_scalar(
                'value_loss/step', self.value_loss, self.step_count)
            self.writer.add_scalar(
                'value_loss/iter', self.value_loss, self.iter_count)
            if len(self.episode_loss_his) > 0:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_loss = self.episode_loss_meter.get_mean()
                mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()
                self.writer.add_scalar(
                    'policy_loss/step', mean_policy_loss, self.step_count)
                self.writer.add_scalar(
                    'policy_loss/time', mean_policy_loss, time_elapse)
                self.writer.add_scalar(
                    'policy_loss/iter', mean_policy_loss, self.iter_count)
                self.writer.add_scalar(
                    'rewards/step', -mean_policy_loss, self.step_count)
                self.writer.add_scalar(
                    'rewards/time', -mean_policy_loss, time_elapse)
                self.writer.add_scalar(
                    'rewards/iter', -mean_policy_loss, self.iter_count)
                self.writer.add_scalar(
                    'policy_discounted_loss/step', mean_policy_discounted_loss, self.step_count)
                self.writer.add_scalar(
                    'policy_discounted_loss/iter', mean_policy_discounted_loss, self.iter_count)
                self.writer.add_scalar(
                    'episode_lengths/iter', mean_episode_length, self.iter_count)
                self.writer.add_scalar(
                    'episode_lengths/step', mean_episode_length, self.step_count)
                self.writer.add_scalar(
                    'episode_lengths/time', mean_episode_length, time_elapse)
            else:
                mean_policy_loss = jnp.inf
                mean_policy_discounted_loss = jnp.inf
                mean_episode_length = 0

            print(
                "iter {}: ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, fps total {:.2f},"
                " value loss {:.2f}, lr {:.6f}".format(
                    self.iter_count,
                    mean_policy_loss,
                    mean_policy_discounted_loss,
                    mean_episode_length,
                    self.steps_num
                    * self.num_envs
                    / (time_end_epoch - time_start_epoch),
                    self.value_loss,
                    lr,
                )
            )

            # update target critic
            def update_target_critic(param, param_targ):
                param_targ *= self.target_critic_alpha
                param_targ += (1.0 - self.target_critic_alpha) * param
                return param_targ

            target_val, critic_def = jax.tree_util.tree_flatten(
                self.target_critic_para)
            critic_val, critic_def = jax.tree_util.tree_flatten(
                self.critic_params)
            for i in range(len(target_val)):
                target_val[i] = update_target_critic(
                    critic_val[i], target_val[i])
            self.target_critic_para = jax.tree_util.tree_unflatten(
                critic_def, target_val
            )

        self.time_report.end_timer("algorithm")
        self.time_report.report()

        # save reward/length history
        self.episode_loss_his = jnp.array(self.episode_loss_his)
        self.episode_discounted_loss_his = jnp.array(
            self.episode_discounted_loss_his)
        self.episode_length_his = jnp.array(self.episode_length_his)
        self.close()

    def save(self, filename=None):
        params = jax.tree_map(lambda x: x, self.actor_params)
        filename = filename + ".pkl"
        with open("./" + filename, "wb") as f:
            pickle.dump(params, f)

    def close(self):
        self.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default="whip_rope")
    parser.add_argument("--num_envs", default=16, type=int)
    parser.add_argument("--actor_lr", default=1e-4, type=float)
    parser.add_argument("--critic_lr", default=1e-4, type=float)
    parser.add_argument("--max_it", default=2000, type=int)
    parser.add_argument("--max_grad_norm", default=0.3, type=float)
    parser.add_argument("--eval_freq", default=20, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument('--gpus', default=1, type=int)

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"{my_path}/cfg/shac.yaml", "r"))

    model = SHAC(cfg, args)
    model.train()
    model.close()
