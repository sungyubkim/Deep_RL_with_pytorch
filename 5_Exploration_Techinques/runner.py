# This code is mainly excerpted from openai baseline code.
# https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py#L2
# https://github.com/openai/baselines/blob/master/baselines/common/runners.py

import numpy as np

from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner
    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, int_gamma, lam, rnd_start=int(1e+3)):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.int_gamma = int_gamma
        self.rnd_start = rnd_start
        self.s_arr = list()

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_int_rewards, mb_actions, mb_values, mb_int_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[], [], []
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, int_values, neglogpacs = self.model.choose_action(self.obs)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_int_values.append(int_values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            if self.model.memory_counter < self.rnd_start:
                # RND state data gather
                for i in range(len(self.obs)):
                    # RND uses only last image
                    self.s_arr.append(self.obs[i, -1])
                r_int = np.zeros_like(rewards)
            elif self.model.memory_counter == self.rnd_start:
                print("RND STAT FINISH")
                # calc state stat
                self.model.s_mu = np.mean(self.s_arr, axis=0)
                self.model.s_sigma = np.std(self.s_arr, axis=0)
                r_int = np.zeros_like(rewards)
            else:
                # get intrinsic reward
                r_int = self.model.r_int(self.obs)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
                    
                   
            mb_rewards.append(rewards)
            mb_int_rewards.append(r_int)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_int_rewards = np.asarray(mb_int_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_int_values = np.asarray(mb_int_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        
        # post processing of intrinsic rewards
        mb_int_rewards = mb_int_rewards / (np.std(mb_int_values) + 1e-8)
        
        # choose action then we get (actions, values, log_probs)
        last_values = self.model.choose_action(self.obs)[1]
        last_int_values = self.model.choose_action(self.obs)[2]

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        
        mb_int_returns = np.zeros_like(mb_int_rewards)
        mb_int_advs = np.zeros_like(mb_int_rewards)
        int_lastgaelam = 0
        
        for t in reversed(range(self.nsteps)):
            
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
                next_intvalues = last_int_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
                next_intvalues = mb_int_values[t+1]
                
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            
            int_delta = mb_int_rewards[t] + self.int_gamma * next_intvalues * nextnonterminal - mb_int_values[t]
            mb_int_advs[t] = int_lastgaelam = int_delta + self.int_gamma * self.lam * nextnonterminal * int_lastgaelam
        
        mb_returns = mb_advs + mb_values
        mb_int_returns = mb_int_advs + mb_int_values
        return (*map(sf01, (mb_obs, mb_returns, mb_int_rewards, mb_int_returns, mb_dones, mb_actions, mb_values, mb_int_values,  mb_neglogpacs)),
            epinfos)
    
    
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])