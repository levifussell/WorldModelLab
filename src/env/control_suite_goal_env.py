from env.goal_env import *

class ControlSuiteGoalEnv(GoalEnv):
    """
    Abstract class.
    Wraps the DeepMind Control Suite gyms.
    """

    def __init__(
            self,
            task_build_func: Callable,
            data_processor: GoalEnvDataProcessor,
            render: bool = False,
            ) -> None:
        """
        :param max_steps: maximium environment steps, usually for testing.
        """
        super().__init__(env_gym=task_build_func(), data_preprocessor=data_processor, render=render)

        self._physics = self._env_gym._physics
        self._task = self._env_gym.task

        self._env_gym._flat_observation = False  

    @property
    def action_size(self):
        return self._env_gym.action_spec().shape[-1]

    @property
    def action_min(self):
        return self._env_gym.action_spec().minimum

    @property
    def action_max(self):
        return self._env_gym.action_spec().maximum

    @property
    def state_size(self):
        state, goal = self.get_curr_global_state_and_goal()
        return state.shape[-1]

    @property
    def goal_size(self):
        state, goal = self.get_curr_global_state_and_goal()
        return goal.shape[-1]

    @property
    def policy_input_size(self):
        state, goal = self.get_curr_global_state_and_goal()
        policy_input = self._data_processor.preprocess_state_and_goal_for_policy(state, goal)
        return policy_input.shape[-1]

    def step(
            self,
            act: torch.Tensor,
            ):

        timestep = self._env_gym.step(act.flatten().numpy()) 

        done = timestep.last()

        global_state, global_goal = self.get_curr_global_state_and_goal()

        info = {}

        if self.render:
            self.frames.append(self._physics.render(camera_id=0, height=200, width=200))

        return global_state, global_goal, done, info

    def reset(self):

        timestep = self._env_gym.reset()

        # done = timestep.last()

        global_state, global_goal = self.get_curr_global_state_and_goal()

        if self.render:
            self.frames.clear()
            self.frames.append(self._physics.render(camera_id=0, height=200, width=200))

        return global_state, global_goal #, done
