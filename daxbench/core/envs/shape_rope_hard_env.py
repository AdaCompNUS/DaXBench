import time

from daxbench.core.envs.others.metric import calc_IOU
from daxbench.core.envs.shape_rope_env import ShapeRopeEnv, DefaultConf
from daxbench.core.utils.util import get_expert_start_end_mpm

ShapeRopeHardConfig = DefaultConf


class ShapeRopeHardEnv(ShapeRopeEnv):

    def __init__(self, batch_size, seed, max_steps=20, conf=None, aux_reward=False):
        super().__init__(batch_size, seed, max_steps=max_steps, conf=conf, aux_reward=aux_reward)

    def reset(self, key):
        super().reset(key)
        self.random_push(step=8)
        return self.get_obs(self.state), self.state


if __name__ == "__main__":

    env = ShapeRopeHardEnv(batch_size=1, seed=0)
    obs, state = env.reset(env.simulator.key)

    print("time start")
    start_time = time.time()
    for i in range(100):
        actions = get_expert_start_end_mpm(state.x, size=512)
        # obs, reward, done, info = env.step_diff(actions, state)
        obs, reward, done, info = env.step_with_render(actions, state)
        state = info["state"]
        print("step", i, state.primitives[0].position[0])
        print("iou", calc_IOU(state.x[0], env.conf.goal_path))
    print(time.time() - start_time)
