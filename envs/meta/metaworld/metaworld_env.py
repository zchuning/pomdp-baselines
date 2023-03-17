import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
from gym import Env

ml45_train = metaworld.ML45(seed=0)
ml45_test = metaworld.ML45(seed=9999)

class MetaWorldEnv(Env):
    def __init__(self, env_name, max_episode_steps=128, n_tasks=51):
        self._max_episode_steps = max_episode_steps
        self.n_tasks = n_tasks

        self.env = ml45_train.train_classes[env_name]()
        self.train_tasks = [task for task in ml45_train.train_tasks if task.env_name == env_name][:50]
        self.test_tasks = [task for task in ml45_test.train_tasks if task.env_name == env_name][:1]
        self.tasks = self.train_tasks + self.test_tasks
        self.reset_task(0)
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def get_train_task_idx(self):
        return list(range(len(self.train_tasks)))

    def get_test_task_idx(self):
        return list(range(len(self.test_tasks)))

    def reset_task(self, idx):
        if idx is not None:
            self.env.set_task(self.tasks[idx])
        self.env.reset()
    
    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def seed(self, seed=None):
        return self.env.seed(seed)

