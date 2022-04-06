from typing import Tuple, Callable, List

import copy
import time

import numpy as np
import torch

from multiprocessing import Process, Pipe, Lock
from multiprocessing.connection import Connection

from ..utils.normalizer import Normalizer

from .buffer import Buffer
from .buffer_iterator import BufferIterator

from ..env.goal_env import GoalEnv, GoalEnvDataProcessor

from ..training.policy import Policy


class EnvCollector:

    P_START_EPISODE_COLLECTION = 0
    P_EPISODE_COMPLETED = 1
    P_NEW_POLICY = 2
    P_WORKER_BUILD_READY = 3

    P_EPISODE_COLLECTION_TYPE_POLICY = 0
    P_EPISODE_COLLECTION_TYPE_RANDOM = 1

    DEBUG_MULTIPROCESSING = False

    def __init__(
            self, 
            f_env_builder: Callable, 
            env_data_processor: GoalEnvDataProcessor,
            policy: Policy,
            policy_kwargs: dict,
            buffer: Buffer,
            min_env_steps: int,
            exploration_std: float,
            normalizer_wm_state: Normalizer,
            normalizer_wm_state_delta: Normalizer,
            normalizer_wm_action: Normalizer,
            normalizer_po_state_and_goal: Normalizer,
            max_env_steps: int = 0,
            collect_device: str = 'cpu',
            train_device: str = 'cuda',
            n_workers: int = 1):
        """
        :param f_env_builder: the method to build the environment to collect from.
        :param buffer: the buffer to store environment samples.
        :param min_env_steps: the minimum size of the window before termination is allowed.
        :param exploration_std: std of the gaussian noise added to the policy.
        :param collect_device:
        :param train_device:
        :param n_workers: number of collection processes. 
        """

        self.env_data_processor = env_data_processor 
        self.buffer = buffer
        self.min_env_steps = min_env_steps
        self.max_env_steps = max_env_steps
        self.exploration_std = exploration_std
        self.collect_device = collect_device
        self.train_device = train_device
        self.n_workers = n_workers

        self.normalizer_wm_state = normalizer_wm_state
        self.normalizer_wm_state_delta = normalizer_wm_state_delta
        self.normalizer_wm_action = normalizer_wm_action
        self.normalizer_po_state_and_goal = normalizer_po_state_and_goal

        self.current_policy = copy.deepcopy(policy).to(self.collect_device)

        # SETUP ENVIRONMENT PROCESSES.

        self.p_env_parent_pipes: List[Connection] = []
        self.p_env_child_pipes: List[Connection] = []

        for w in range(self.n_workers):

            conn_parent, conn_child = Pipe()
            self.p_env_parent_pipes.append(conn_parent)
            self.p_env_child_pipes.append(conn_child)

        self.p_envs = []
        for w in range(self.n_workers):

            p_env = Process(
                        target=EnvCollector._launch_collection_process, 
                        args=(
                            w,
                            self.p_env_child_pipes[w],
                            f_env_builder(),
                            env_data_processor,
                            # copy.deepcopy(policy),
                            policy_kwargs,
                            collect_device,
                            exploration_std,
                            min_env_steps,
                            max_env_steps,
                        ))
            p_env.start()
            self.p_envs.append(p_env)

    def copy_current_policy(self, policy: Policy):
        """
        Copies a policy to the gym's policy
        """

        # TODO: need a lock here so we don't write/read at same time.

        # self.current_policy = copy.deepcopy(policy).to(self.collect_device)

        # if self.current_policy is None:
        #     self.current_policy = copy.deepcopy(policy).to(self.collect_device)
        # else:
            # self.current_policy.load_state_dict(policy.state_dict())

        pol_device = policy.device
        policy.to(self.collect_device)

        self.current_policy.load_state_dict(policy.state_dict())

        for w in range(self.n_workers):
            self.p_env_parent_pipes[w].send([
                EnvCollector.P_NEW_POLICY,
                policy.state_dict(),
            ])

        policy.to(pol_device)

    # def start_gym_process(self):
    #     """
    #     Launches a gym in a separate process and samples from it.
    #     """

    #     # TODO: launch a separate process with the gym and collect it in the buffer.

    #     if not self.is_parallel:
    #         raise Exception("Collect is in parallel mode.")

    #     pass

    def sample_buffer(
        self,
        nsamples: int,
        batchsize: int,
        window_size: int
    ) -> torch.tensor:
        """
        Returns an iterator that gives batches for samples from a buffer.

        :param nsamples: number of samples from the buffer.
        :param batchsize: size of the minibatches in the samples.
        :param window_size: size of the windows in the samples.
        :return: iterator for batches of shape (N, T, F)
        """

        # TODO: need a lock here so we don't write/read at same time.

        data_sample = self.buffer.sample(
            nsamples=nsamples,
            window_size=window_size,
        )

        return BufferIterator(
            data=data_sample,
            batchsize=batchsize,
            device=self.train_device,
        )

    def warmup_normalizer(self, warmup_steps) -> int:

        n_steps = 0

        while n_steps < warmup_steps:

            done = False

            states = []
            dstates = []
            goals = []
            acts = []

            state, goal = self.env.reset()

            states.append(state)
            goals.append(goal)

            while True:

                act = torch.randn(self.current_policy.action_size) * self.exploration_std
                acts.append(act)
                
                next_state, goal, done, info = self.env.step(act)

                dstates.append(self.env.compute_delta_state_world_model(state_from=state, state_to=next_state))

                # TODO: environment specific. Check the environment termination behaviour.
                if done and len(states) >= self.min_env_steps:
                    break

                # TODO: environment specific. Check the environment termination behaviour.
                if self.max_env_steps > 0 and len(states) >= self.max_env_steps:
                    done = True # TODO: make this a case-specific done.
                    break

                states.append(next_state)
                goals.append(goal)

                state = next_state

            n_steps += len(states)

        states = torch.cat([s.reshape(1,-1) for s in states], dim=0)
        dstates = torch.cat([s.reshape(1,-1) for s in dstates], dim=0)
        goals = torch.cat([g.reshape(1,-1) for g in goals], dim=0)
        acts = torch.cat([a.reshape(1,-1) for a in acts], dim=0)

        states_and_goals = self.env.preprocess_state_and_goal_for_policy(state=states, goal=goals)
        states = self.env.preprocess_state_for_world_model(states)

        self.normalizer_wm_state.warmup(states)
        self.normalizer_wm_state_delta.warmup(dstates)
        self.normalizer_wm_action.warmup(acts)
        self.normalizer_po_state_and_goal.warmup(states_and_goals)

    @staticmethod
    def _launch_collection_process(
        id: int,
        conn: Connection,
        env: GoalEnv,
        env_data_processor: GoalEnvDataProcessor,
        policy_kwargs: dict,
        collect_device: str,
        exploration_std: float,
        min_env_steps: int,
        max_env_steps: int,
    ):
        """
        Runs a collection process that will collect episodes and push them to the buffer.

        Args:
            id (int): ID of the process.
            conn (Connection): child connection of the Pipe.
            f_env_builder (Callable): function to build the environment.
            policy (Policy): policy to collect from environment.
            buffer (Buffer): buffer to add the data to.
            buffer_lock (Lock): lock for adding to the buffer.
        """

        # BUILD POLICY.

        policy = Policy(
                fn_combine_state_and_goal=env_data_processor.preprocess_state_and_goal_for_policy,
                fn_post_process_action=lambda x : x,
                **policy_kwargs,
                ).to(collect_device)

        # PROCESS PARAMETERS.

        wait_for_episode_collection = True

        use_policy = True

        # if EnvCollector.DEBUG_MULTIPROCESSING:
        print(f"PROCESS {id} CREATED.")

        # SEND WORKER READY CALL.

        conn.send([EnvCollector.P_WORKER_BUILD_READY, None])

        # RUN THE COLLECTION.

        while True:

            time.sleep(0.0001)

            try:
                control_msg, control_data = conn.recv()
            except (EOFError, KeyboardInterrupt):
                break

            # CHECK FOR NEW POLICY.

            if control_msg == EnvCollector.P_NEW_POLICY:
                policy.load_state_dict(control_data)

                if EnvCollector.DEBUG_MULTIPROCESSING:
                    print(f"PROCESS {id} NEW POLICY RECEIVED.")

            # CHECK IF EPISODE CAN BE COLLECTED.

            if control_msg == EnvCollector.P_START_EPISODE_COLLECTION:
                wait_for_episode_collection = False

                if control_data == EnvCollector.P_EPISODE_COLLECTION_TYPE_POLICY:
                    use_policy = True
                elif control_data == EnvCollector.P_EPISODE_COLLECTION_TYPE_RANDOM:
                    use_policy = False
                else:
                    raise Exception("Invalid collection type.")

            if wait_for_episode_collection:
                continue

            # COLLECT EPISODE.

            if EnvCollector.DEBUG_MULTIPROCESSING:
                print(f"PROCESS {id} STARTING COLLECTION.")

            done = False

            states = []
            goals = []
            acts = []
            rewards = []

            state, goal = env.reset()

            states.append(state)
            goals.append(goal)

            while True:

                with torch.no_grad():

                    if use_policy:

                        act = policy(
                                    state=state.to(collect_device),
                                    goal=goal.to(collect_device),
                                    ).cpu()

                        act += torch.randn(act.shape).to(act.device) * exploration_std

                    else:

                        act = torch.randn(policy.action_size) * exploration_std

                acts.append(act)

                next_state, next_goal, done, info = env.step(act)

                    # NOTE: we want to compare the loss of the NEXT state with the PREVIOUS goal.
                rewards.append(env_data_processor.compute_reward(state=next_state, goal=goal, act=act))

                # TODO: environment specific. Check the environment termination behaviour.
                if done and len(states) >= min_env_steps:
                    break

                # TODO: environment specific. Check the environment termination behaviour.
                if max_env_steps > 0 and len(states) >= max_env_steps:
                    done = True # TODO: make this a case-specific done.
                    break

                states.append(state)

                goals.append(goal)

                state = next_state
                goal = next_goal

            states = torch.cat([s.reshape(1,-1) for s in states], dim=0)
            goals = torch.cat([g.reshape(1,-1) for g in goals], dim=0)
            acts = torch.cat([a.reshape(1,-1) for a in acts], dim=0)

            reward_sum = np.sum(rewards)

            n_steps = len(states)

            if EnvCollector.DEBUG_MULTIPROCESSING:
                print(f"PROCESS {id} FINISHED COLLECTION.")

            # FLAG TO WAIT FOR PARENT TO PROCESS THE DATA

            wait_for_episode_collection = True

            # SEND THE DATA TO PARENT.

            conn.send([
                EnvCollector.P_EPISODE_COMPLETED, 
                dict(
                    states=states,
                    goals=goals,
                    acts=acts,
                    reward_sum=reward_sum,
                    n_steps=n_steps,
                )
            ])

        if EnvCollector.DEBUG_MULTIPROCESSING:
            print(f"PROCESS {id} TERMINATED.")
            
    def collect_nsteps(
        self,
        min_num_steps: int,
        use_policy: bool = True,
        update_normalizers: bool = False,
    ) -> Tuple[int, int, torch.Tensor]:

        n_total_steps = 0
        n_total_episodes = 0
        returns = []

        # START THE COLLECTION PROCESS.

        while n_total_steps < min_num_steps:

            for w in range(self.n_workers):

                child_msg, child_data = self.p_env_parent_pipes[w].recv()

                if child_msg == EnvCollector.P_WORKER_BUILD_READY:

                    if use_policy:
                        self.p_env_parent_pipes[w].send([EnvCollector.P_START_EPISODE_COLLECTION, EnvCollector.P_EPISODE_COLLECTION_TYPE_POLICY])
                    else:
                        self.p_env_parent_pipes[w].send([EnvCollector.P_START_EPISODE_COLLECTION, EnvCollector.P_EPISODE_COLLECTION_TYPE_RANDOM])

                elif child_msg == EnvCollector.P_EPISODE_COMPLETED:

                    # add data to buffer.

                    self.buffer.add(
                        state=child_data['states'],
                        goal=child_data['goals'],
                        act=child_data['acts'],
                    )

                    returns.append(child_data['reward_sum'])
                    n_total_steps += child_data['n_steps']
                    n_total_episodes += 1

                    if EnvCollector.DEBUG_MULTIPROCESSING:
                        print(f"PROCESS {w} COLLECTED {child_data['n_steps']} STEPS")
                        print(f"COLLECTED TOTAL {n_total_steps} STEPS")

                    # update normalizers.

                    if update_normalizers:

                        self.normalizer_wm_state += self.env_data_processor.preprocess_state_for_world_model(state=child_data['states'])
                        self.normalizer_wm_action += child_data['acts']
                        # self.normalizer_wm_state_delta += self.env_data_processor.compute_delta_state_world_model(
                        #                                                                 state_from=child_data['states'][:-1], 
                        #                                                                 state_to=child_data['states'][1:],
                        #                                                                 )
                        self.normalizer_wm_state_delta += child_data['states']

                        self.normalizer_po_state_and_goal += self.env_data_processor.preprocess_state_and_goal_for_policy(
                                                                                        state=child_data['states'], 
                                                                                        goal=child_data['goals'],
                                                                                        )

                    # restart collection.

                    # NOTE: this will mean that the collection happens during training, and the policy will be slightly behind,
                    #  but it should be a small effect.
                    # COMMENTED OUT TO ENABLE THIS: if n_total_steps < min_num_steps:

                    if use_policy:
                        self.p_env_parent_pipes[w].send([EnvCollector.P_START_EPISODE_COLLECTION, EnvCollector.P_EPISODE_COLLECTION_TYPE_POLICY])
                    else:
                        self.p_env_parent_pipes[w].send([EnvCollector.P_START_EPISODE_COLLECTION, EnvCollector.P_EPISODE_COLLECTION_TYPE_RANDOM])

            time.sleep(0.1)

        return n_total_steps, n_total_episodes, torch.FloatTensor(returns)


    # def collect_nsteps(
    #     self,
    #     min_num_steps: int,
    # ) -> Tuple[int, int, torch.Tensor]:

    #     n_total_steps = 0
    #     n_total_episodes = 0
    #     returns = []

    #     # # LAUNCH THE PROCESSES COLLECTING.

    #     # for w in range(self.n_workers):

    #     #     self.p_env_parent_pipes[w].send([EnvCollector.P_START_EPISODE_COLLECTION, None])
        
    #     # START THE COLLECTION PROCESS.

    #     while n_total_steps < min_num_steps:

    #         for w in range(self.n_workers):

    #             child_msg, child_data = self.p_env_parent_pipes[w].recv()

    #             if child_msg == EnvCollector.P_WORKER_BUILD_READY:

    #                 self.p_env_parent_pipes[w].send([EnvCollector.P_START_EPISODE_COLLECTION, None])

    #             elif child_msg == EnvCollector.P_EPISODE_COMPLETED:

    #                 # add data to buffer.

    #                 self.buffer.add(
    #                     state=child_data['states'],
    #                     goal=child_data['goals'],
    #                     act=child_data['acts'],
    #                 )

    #                 returns.append(child_data['reward_sum'])
    #                 n_total_steps += child_data['n_steps']
    #                 n_total_episodes += 1

    #                 if EnvCollector.DEBUG_MULTIPROCESSING:
    #                     print(f"PROCESS {w} COLLECTED {child_data['n_steps']} STEPS")
    #                     print(f"COLLECTED TOTAL {n_total_steps} STEPS")

    #                 # update normalizers.

    #                 self.normalizer_wm_state += self.env_data_processor.preprocess_state_for_world_model(state=child_data['states'])
    #                 self.normalizer_wm_action += child_data['acts']
    #                 self.normalizer_wm_state_delta += self.env_data_processor.compute_delta_state_world_model(
    #                                                                                 state_from=child_data['states'][:-1], 
    #                                                                                 state_to=child_data['states'][1:],
    #                                                                                 )

    #                 self.normalizer_po_state_and_goal += self.env_data_processor.preprocess_state_and_goal_for_policy(
    #                                                                                 state=child_data['states'], 
    #                                                                                 goal=child_data['goals'],
    #                                                                                 )

    #                 # restart collection.

    #                 # NOTE: this will mean that the collection happens during training, and the policy will be slightly behind,
    #                 #  but it should be a small effect.
    #                 # if n_total_steps < min_num_steps:
    #                 self.p_env_parent_pipes[w].send([EnvCollector.P_START_EPISODE_COLLECTION, None])

    #         time.sleep(0.1)

    #     return n_total_steps, n_total_episodes, torch.FloatTensor(returns)

    # def collect(
    #     self,
    #     min_num_steps: int,
    # ) -> Tuple[int, int, torch.Tensor]:
    #     """
    #     Manually collects from the environment. Used for non-parallel training.

    #     :param min_num_steps: minimum number of steps for the policy to collect.
    #     """

    #     if self.is_parallel:
    #         raise Exception("Collector is parallel. Shouldn't be doing manual collecting.")

    #     n_steps = 0
    #     n_eps = 0

    #     returns = []

    #     render_frames = []

    #     while n_steps < min_num_steps:

    #         done = False

    #         states = []
    #         goals = []
    #         acts = []
    #         rewards = []

    #         state, goal = self.env.reset()

    #         states.append(state)
    #         if self.normalizer_wm_state is not None:
    #             self.normalizer_wm_state += self.env.preprocess_state_for_world_model(state)

    #         goals.append(goal)

    #         state_and_goal = self.env.preprocess_state_and_goal_for_policy(state=state, goal=goal)
    #         if self.normalizer_po_state_and_goal is not None:
    #             self.normalizer_po_state_and_goal += state_and_goal

    #         while True:

    #             with torch.no_grad():

    #                 act = self.current_policy(
    #                             state=state.to(self.collect_device),
    #                             goal=goal.to(self.collect_device),
    #                             ).cpu()

    #                 act += torch.randn(act.shape).to(act.device) * self.exploration_std

    #             acts.append(act)
    #             if self.normalizer_wm_action is not None:
    #                 self.normalizer_wm_action += act

    #             rewards.append(self.env.reward(state=state, goal=goal, act=act))

    #             next_state, goal, done, info = self.env.step(act)

    #             if self.normalizer_wm_state_delta is not None:
    #                 self.normalizer_wm_state_delta += self.env.compute_delta_state_world_model(state_from=state, state_to=next_state)

    #             # TODO: environment specific. Check the environment termination behaviour.
    #             if done and len(states) >= self.min_env_steps:
    #                 break

    #             # TODO: environment specific. Check the environment termination behaviour.
    #             if self.max_env_steps > 0 and len(states) >= self.max_env_steps:
    #                 done = True # TODO: make this a case-specific done.
    #                 break

    #             states.append(next_state)
    #             if self.normalizer_wm_state is not None:
    #                 self.normalizer_wm_state += self.env.preprocess_state_for_world_model(next_state)

    #             goals.append(goal)

    #             state_and_goal = self.env.preprocess_state_and_goal_for_policy(state=next_state, goal=goal)
    #             if self.normalizer_po_state_and_goal is not None:
    #                 self.normalizer_po_state_and_goal += state_and_goal

    #             state = next_state

    #         self.buffer.add(
    #             state=torch.cat([s.reshape(1,-1) for s in states], dim=0),
    #             goal=torch.cat([g.reshape(1,-1) for g in goals], dim=0),
    #             act=torch.cat([a.reshape(1,-1) for a in acts], dim=0),
    #         )

    #         returns.append(np.sum(rewards))

    #         if self.env.render:
    #             render_frames.extend(self.env.frames)

    #         n_steps += len(states)
    #         n_eps += 1

    #     return n_steps, n_eps, torch.FloatTensor(returns), render_frames