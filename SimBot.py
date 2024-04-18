import random
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class Spell:
    def __init__(self, name, cooldown, damage):
        self.name = name
        self.cooldown = cooldown
        self.current_cooldown = 0
        self.damage = damage

    def cast(self):
        if self.current_cooldown == 0:
            self.current_cooldown = self.cooldown
            print(f"Casting {self.name}, setting cooldown to {self.cooldown}")
            return True
        print(f"Cannot cast {self.name}, cooldown remaining: {self.current_cooldown}")
        return False

    def cooldown_tick(self):
        if self.current_cooldown > 0:
            self.current_cooldown -= 1


class Fireball(Spell):
    def __init__(self, name, cooldown, damage, training_dummy):
        super().__init__(name, cooldown, damage)
        self.training_dummy = training_dummy

    def cast(self):
        if super().cast():
            self.training_dummy.calc_damage(self.damage)
            return True
        return False


class Frostbolt(Spell):
    def __init__(self, name, cooldown, damage, training_dummy):
        super().__init__(name, cooldown, damage)
        self.training_dummy = training_dummy

    def cast(self):
        if super().cast():
            self.training_dummy.calc_damage(self.damage)
            return True
        return False


class BloodMoonCrescent(Spell):
    def __init__(self, name, cooldown, damage, training_dummy):
        super().__init__(name, cooldown, damage)
        self.training_dummy = training_dummy

    def cast(self):
        if super().cast():
            self.training_dummy.calc_damage(self.damage)
            return True
        return False


class Blaze(Spell):
    def __init__(self, name, cooldown, damage, training_dummy, character):
        super().__init__(name, cooldown, damage)
        self.training_dummy = training_dummy
        self.character = character

    def cast(self):
        if super().cast():
            if self.character.last_spell == "Fireball":
                applied_damage = self.damage * 5
                self.training_dummy.calc_damage(applied_damage)
                print(f"Blaze cast after Fireball, damage applied: {applied_damage}")
            else:
                self.training_dummy.calc_damage(self.damage)
                print(f"Blaze cast without Fireball, damage applied: {self.damage}")
            return True
        return False


class ScorchDot(Spell):
    def __init__(self, name, cooldown, duration, damage, training_dummy):
        super().__init__(name, cooldown, damage)
        self.duration = duration
        self.training_dummy = training_dummy

    def cast(self):
        if super().cast():
            self.training_dummy.apply_dot(self.damage, self.duration)
            return True
        return False


class Combustion(Spell):
    def __init__(self, name, cooldown, duration, damage_increase, character):
        super().__init__(name, cooldown, duration)
        self.duration = duration
        self.character = character
        self.damage_increase = damage_increase

    def cast(self):
        if super().cast():
            self.character.activate_buff(self.name, self.duration, self.damage_increase)
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
        self.buff_damage_increase = None
        self.buff_name = None
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


class RunSimEnv(gym.Env):
    def __init__(self):
        super(RunSimEnv, self).__init__()
        self.tick_count = 0
        self.env = RunSim()
        # Action space: one discrete action per spell
        self.action_space = spaces.Discrete(len(self.env.spells))
        # Observation space: assuming some max values for illustration
        max_cooldown = 60
        max_damage = 2000
        max_duration = 20
        num_features = 1 + 2 * len(self.env.spells)  # total_damage + cd+count*2
        self.observation_space = spaces.Box(
            # low=np.array([0] * len(self.env.spells) * 2 + [0, 0]),
            # high=np.array([max_cooldown] * len(self.env.spells) + [max_damage] * len(self.env.spells) + [max_duration, max_duration]),
            low=np.float32(0),
            high=np.float32(np.inf),
            shape=(num_features,),
            dtype=np.float32
        )

    def step(self, action):
        action_name = list(self.env.spells.keys())[action]
        self.env.step(action_name)
        state = self.env.get_results()  # TODO: Adjust get_results to return a flat array
        reward = state[2]  # Total-damage in numpy get_results()
        done = self.tick_count >= 128
        self.tick_count += 1  # Increment tick count for Done
        info = {}
        return state, reward, done, info

    def reset(self):
        self.tick_count = 0  # Reset tick count
        return self.env.reset()

    def render(self, mode='human'):
        self.env.render()


class RunSim:
    def __init__(self):
        self.character = Character()
        self.training_dummy = TrainingDummy(self.character)    # Because the trainingsdummy must know,
        # List of spells                                            # the buff of the player
        self.spells = {
            'Fireball': Fireball
            ('Fireball', 0, 10, self.training_dummy),               # Name, Cooldown, Damage

            'Frostbolt': Frostbolt
            ('Frostbolt', 0, 3, self.training_dummy),               # Name, Cooldown, Damage

            'BloodMoonCrescent': BloodMoonCrescent
            ('BloodMoonCrescent', 10, 80, self.training_dummy),     # Name, Cooldown, Damage

            'Blaze': Blaze
            ('Blaze', 0, 5, self.training_dummy, self.character),   # Name, Cooldown, Damage

            'ScorchDoT': ScorchDot
            ('ScorchDoT', 0, 20, 5, self.training_dummy),           # Name, Cooldown, Duration, Damage

            'Combustion': Combustion
            ('Combustion', 60, 25, 0.5, self.character)             # Name, Cooldown, Duration, Damage_increase
        }
        self.spell_cast_count = {name: 0 for name in self.spells.keys()}

    # reset-function for Reinforcement Learning
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
        if spell.cast():
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

    def get_results(self):
        cooldowns = [spell.current_cooldown for _, spell in self.spells.items()]
        cast_counts = [self.spell_cast_count[name] for name in self.spells.keys()]
        state = [self.training_dummy.damage_taken] + cooldowns + cast_counts
        return np.array(state, dtype=np.float32)

    def simulate(self, ticks_amount):
        for _ in range(ticks_amount):  # Simulate Ticks/Seconds
            valid_actions = [name for name, spell in self.spells.items() if spell.current_cooldown == 0]
            chosen_spell = random.choice(valid_actions)
            self.step(chosen_spell)


def main():
    # Create the environment
    env = DummyVecEnv([lambda: RunSimEnv()])

    # Initialize and train the model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)  # Adjust total timesteps according to needs

    # Save the model
    model.save("ppo_runsim")

    # Optionally, you can reload the model
    # model = PPO.load("ppo_runsim", env=env)

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation: mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment when done
    env.close()


if __name__ == "__main__":
    main()

# TODO: random zu ML ausbauen
