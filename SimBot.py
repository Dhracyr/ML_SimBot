import os
import time

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from IPython.display import display, clear_output


class Spell:
    def __init__(self, name, cooldown, damage):
        self.name = name
        self.cooldown = cooldown
        self.current_cooldown = 0
        self.damage = damage

    def cast(self, run_sim):
        if self.current_cooldown == 0:
            self.current_cooldown = self.cooldown
            # print(f"Casting {self.name}, setting cooldown to {self.cooldown}")
            return True
        # print(f"Cannot cast {self.name}, cooldown remaining: {self.current_cooldown}")
        return False

    def cooldown_tick(self):
        if self.current_cooldown > 0:
            self.current_cooldown -= 1


class Fireball(Spell):

    def cast(self, run_sim):
        if super().cast(run_sim):
            training_dummy = run_sim.training_dummy
            training_dummy.calc_damage(self.damage)
            return True
        return False


class Frostbolt(Spell):
    def cast(self, run_sim):
        if super().cast(run_sim):
            training_dummy = run_sim.training_dummy
            training_dummy.calc_damage(self.damage)
            return True
        return False


class BloodMoonCrescent(Spell):

    def cast(self, run_sim):
        if super().cast(run_sim):
            training_dummy = run_sim.training_dummy
            training_dummy.calc_damage(self.damage)
            return True
        return False


class Blaze(Spell):
    def cast(self, run_sim):
        if super().cast(run_sim):
            character = run_sim.character
            training_dummy = run_sim.training_dummy
            if character.last_spell == "Fireball":
                applied_damage = self.damage * 5
                training_dummy.calc_damage(applied_damage)
                # print(f"Blaze cast after Fireball, damage applied: {applied_damage}")
            else:
                training_dummy.calc_damage(self.damage)
                # print(f"Blaze cast without Fireball, damage applied: {self.damage}")
            return True
        return False


class ScorchDot(Spell):
    def __init__(self, name, cooldown, duration, damage):
        super().__init__(name, cooldown, damage)
        self.duration = duration

    def cast(self, run_sim):
        if super().cast(run_sim):
            training_dummy = run_sim.training_dummy
            training_dummy.apply_dot(self.damage, self.duration)
            return True
        return False


class Combustion(Spell):
    def __init__(self, name, cooldown, duration, damage_increase):
        super().__init__(name, cooldown, duration)
        self.duration = duration
        self.damage_increase = damage_increase

    def cast(self, run_sim):
        if super().cast(run_sim):
            # training_dummy = run_sim.training_dummy
            character = run_sim.character
            character.activate_buff(self.name, self.duration, self.damage_increase)
            return True
        return False


class TrainingDummy:

    def __init__(self, character):
        self.dot_damage = 0
        self.dot_timer = 0
        self.damage_taken = 0
        self.character = character

    def apply_dot(self, dot_damage, duration):
        self.dot_timer = duration
        self.dot_damage = dot_damage

    def tick(self):
        # Check Dot-Timer
        if self.dot_timer > 0:
            self.dot_timer -= 1
            self.calc_damage(self.dot_damage)

    def calc_damage(self, damage):
        if self.character.buff_active:
            self.damage_taken += damage * (1+self.character.buff_damage_increase)
        else:
            self.damage_taken += damage


class Character:
    def __init__(self):
        self.buff_damage_increase = 0.0
        self.buff_name = ""
        self.buff_active = False
        self.buff_timer = 0
        self.last_spell = None

    def activate_buff(self, name, duration, damage_increase):
        self.buff_active = True
        self.buff_timer = duration
        self.buff_name = name
        self.buff_damage_increase = damage_increase

    # Check Buff-Timer
    def tick(self):
        if self.buff_timer > 0:
            self.buff_timer -= 1
            if self.buff_timer == 0:
                self.buff_active = False


class RunSim:
    training_dummy = None

    def __init__(self):
        self.character = Character()
        self.training_dummy = TrainingDummy(self.character)    # Because the trainingsdummy must know,
        # List of spells                                            # the buff of the player
        self.spells = {

            'Fireball': Fireball('Fireball', 0, 10),                                # Name, Cooldown, Damage
            'Frostbolt': Frostbolt('Frostbolt', 0, 3),                              # Name, Cooldown, Damage
            'BloodMoonCrescent': BloodMoonCrescent('BloodMoonCrescent', 10, 80),    # Name, Cooldown, Damage
            'Blaze': Blaze('Blaze', 0, 5),                                          # Name, Cooldown, Damage
            'ScorchDoT': ScorchDot('ScorchDoT', 0, 20, 5),                          # Name, Cooldown, Duration, Damage
            'Combustion': Combustion('Combustion', 60, 25, 0.5)                     # Name, Cooldown, Duration, Damage_increase
        }
        self.spell_cast_count = {name: 0 for name in self.spells.keys()}

    # Reset-function for Reinforcement Learning
    def reset(self):
        self.character = Character()
        self.training_dummy = TrainingDummy(self.character)
        for spell in self.spells.values():
            spell.current_cooldown = 0
        self.spell_cast_count = {name: 0 for name in self.spells.keys()}
        return self.get_results()

    def step(self, action_name):
        # Check last_casted_spell & increase cast_count
        spell = self.spells[action_name]
        if spell.cast(self):
            self.character.last_spell = spell.name
            self.spell_cast_count[action_name] += 1

        # Check Dot-Spell/Buff Tick
        self.training_dummy.tick()
        self.character.tick()

        # Check Cooldown Tick
        for spell in self.spells.values():
            spell.cooldown_tick()

        # Create output for visualisation
        self.render()

    def render(self):
        """
        print(f"Total Damage: {self.training_dummy.damage_taken} damage")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"Remaining DoT Duration: {self.training_dummy.dot_timer}")
        print(f"Remaining Buff Duration: {self.character.buff_timer}. "
              f" CD-Buff: {self.spells['Combustion'].current_cooldown}."
              f" CD-Moon: {self.spells['BloodMoonCrescent'].current_cooldown}")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"Spell cast last time: {self.character.last_spell}")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        """
    def get_results(self):
        cooldowns = [spell.current_cooldown for _, spell in self.spells.items()]
        cast_counts = [self.spell_cast_count[name] for name in self.spells.keys()]
        state = [self.training_dummy.damage_taken] + cooldowns + cast_counts
        # print(f"Current state being returned: {state}")
        return np.array(state, dtype=np.float32)

    """
    def simulate(self, ticks_amount):
        for _ in range(ticks_amount):  # Simulate Ticks/Seconds
            valid_actions = [name for name, spell in self.spells.items() if spell.current_cooldown == 0]
            chosen_spell = random.choice(valid_actions)
            self.step(chosen_spell)
    """


def run_simulation():
    class RunSimEnv(gym.Env):
        def __init__(self):
            super(RunSimEnv, self).__init__()
            self.tick_count = 0
            self.env = RunSim()
            # Action space: one discrete action per spell
            self.action_space = spaces.Discrete(len(self.env.spells))
            # Observation space: assuming some max values for illustration
            max_cooldown = 60
            max_damage = 3565
            max_count = 128
            num_features = 1 + 2 * len(self.env.spells)  # total_damage + cd*6 (foreach spell)
            self.observation_space = spaces.Box(
                low=np.zeros(num_features, dtype=np.float32),  # All lows are 0
                high=np.array([max_damage] + ([max_count]+[max_cooldown]) * len(self.env.spells)),
                # low=np.float32(0),
                # high=np.float32(np.inf),
                # shape=(num_features,),
                dtype=np.float32
            )

        def step(self, action):
            action_name = list(self.env.spells.keys())[action]
            self.env.step(action_name)
            obs = self.env.get_results()
            reward = obs[0]  # Total-damage in numpy get_results()
            done = self.tick_count >= 128
            self.tick_count += 1  # Increment tick count for Done
            info = {}
            return obs, reward, done, info

        def reset(self):
            self.tick_count = 0  # Reset tick count
            return self.env.reset()

        def render(self, mode='human'):
            self.env.render()

    # Initialize the custom environment
    env = DummyVecEnv([lambda: RunSimEnv()])
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # This is specific to certain OS environments

    # Set up plot for live updating
    fig, ax = plt.subplots()
    ax.set_xlabel('Ticks')
    ax.set_ylabel('Total Damage')  # Max-Damage possible: 3565 in 128-Ticks
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 3565)
    plt.ion()  # Turn on interactive mode

    # Initialize & train the RL model
    model = PPO("MlpPolicy", env, verbose=1, gamma=0.9, n_steps=2048, ent_coef=0.01, batch_size=2048, gae_lambda=0.95).learn(total_timesteps=5000000)

    # Save the model
    save_dir = "/tmp/gym/"
    os.makedirs(save_dir, exist_ok=True)
    model.save(f"{save_dir}/ppo_runsim")

    # Optionally, you can reload it
    # model = PPO.load(f"{save_dir}/ppo_runsim", env=env, verbose=1)
    # show the save hyperparameters
    # print(f"loaded: gamma={model.gamma}, n_steps={model.n_steps}")
    # as the environment is not serializable, we need to set a new instance of the environment
    # model.set_env(DummyVecEnv([lambda: RunSimEnv()]))
    # model.learn(8000)

    tick_count = 0
    tick_data = []
    reward_data = []

    max_damage = 0
    min_damage = 3000
    steps_until_stop = 8000
    for i in range(steps_until_stop):  # Number of steps or until a stopping criterion
        action = [env.action_space.sample()]  # Random action or from model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        # Manage the plot update
        tick_data.append(tick_count)
        reward_data.append(rewards[0])

        if i % 1 == 0:
            plot_color = 'purple'
            if i >= 8000:
                plot_color = 'blue'
            elif i >= 7999:
                plot_color = 'green'
            elif i >= 6000:
                plot_color = 'yellow'
            elif i >= 4000:
                plot_color = 'orange'
            elif i >= 2000:
                plot_color = 'red'
            else:
                plot_color = 'black'
        ax.plot(tick_data, reward_data, color=plot_color)
        display(plt.gcf())
        clear_output(wait=True)

        if dones[0]:
            tick_count = 0
            tick_data = []
            reward_data = []
        else:
            tick_count += 1
            if rewards[0] > max_damage:
                max_damage = rewards[0]
                max_stats = [obs[0][7], obs[0][8], obs[0][9], obs[0][10], obs[0][11], obs[0][12]]
            if rewards[0] < max_damage:
                min_damage = rewards[0]
                min_stats = [obs[0][7], obs[0][8], obs[0][9], obs[0][10], obs[0][11], obs[0][12]]
    plt.ioff()  # Turn off interactive mode
    plt.show()

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"Evaluation: mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # print best one
    if None not in max_stats:
        print(max_damage, "of 3565")
        print(f"Fireball used: {max_stats[0]}")
        print(f"Frostbolt used: {max_stats[1]}")
        print(f"BloodMoonCrescent used: {max_stats[2]}")
        print(f"Blaze used: {max_stats[3]}")
        print(f"ScorchDot used: {max_stats[4]}")
        print(f"Combustion used: {max_stats[5]}")
    else:
        print("fuck you")

    print("------"*5)
    print("------"*5)
    # print worst one
    if None not in min_stats:
        print(min_damage, "of 3565")
        print(f"Fireball used: {min_stats[0]}")
        print(f"Frostbolt used: {min_stats[1]}")
        print(f"BloodMoonCrescent used: {min_stats[2]}")
        print(f"Blaze used: {min_stats[3]}")
        print(f"ScorchDot used: {min_stats[4]}")
        print(f"Combustion used: {min_stats[5]}")
    else:
        print("fuck you")

    env.close()


def main():
    pass


if __name__ == "__main__":
    run_simulation()
    # main()

# TODO: Ist input tatsächlich nur random???
# TODO: Ab wann ist overfitting
# TODO: Parameter durch Cross-Entropy versuchen
# TODO: Datenbank auslagern?
# TODO: Mal bissl aufräumen...
# TODO: Fuck es ist Halb 4...
# TODO: Docu nochmal anschaun/Jupyter Notebooks

"""
6 Stages of Success:
    1. Stop casting spells, that are on cooldown
    2. Stop casting Frostbolt, because its useless
    3. Stop casting DoT, when its already active
    4. Casting Combustion and BloodMoon on Cooldown, because it does most damage
    5. Alternating between Fireball and Blaze
    6. Waiting for BloodMoon cooldown when buff-cd is going to expire soon
"""
