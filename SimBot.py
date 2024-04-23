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
            'ScorchDot': ScorchDot('ScorchDot', 0, 20, 5),                          # Name, Cooldown, Duration, Damage
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
        cooldowns = [spell.current_cooldown / 60 for _, spell in self.spells.items()]
        cast_counts = [self.spell_cast_count[name] / 128 for name in self.spells.keys()]
        last_spell = [1] if self.character.last_spell == 'Blaze' else [0]
        state = [self.training_dummy.damage_taken] + cooldowns + cast_counts + last_spell
        # print(f"Current state being returned: {state}")
        return np.array(state, dtype=np.float32)

    # Old simulate
    """
    def simulate(self, ticks_amount):
        for _ in range(ticks_amount):  # Simulate Ticks/Seconds
            valid_actions = [name for name, spell in self.spells.items() if spell.current_cooldown == 0]
            chosen_spell = random.choice(valid_actions)
            self.step(chosen_spell)
    """


def draw_plot(env, model, ax, obs):
    tick_count = 0
    tick_data = []
    reward_data = []

    max_damage = 0
    max_stats = []

    for i in range(128):
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        # Manage the plot update
        tick_data.append(tick_count)
        reward_data.append(rewards)

        ax.set_xlabel('Ticks')
        ax.set_ylabel('Total Damage')  # Max-Damage possible: 4242.5 in 128-Ticks
        ax.set_xlim(0, 128)
        ax.set_ylim(0, global_max_damage)  # 4242.5
        ax.plot(tick_data, reward_data, color='red', alpha=0.8)
        display(plt.gcf())
        # clear_output(wait=True)

        if dones:
            obs = env.reset()
            tick_count = 0
            tick_data = []
            reward_data = []
        else:
            tick_count += 1
            if rewards > max_damage:
                max_damage = rewards
                max_stats = obs[7]*128, obs[8]*128, obs[9]*128, obs[10]*128, obs[11]*128, obs[12]*128
            # if rewards < max_damage:
            # min_damage = rewards
            # min_stats = obs[7]*128, obs[8]*128, obs[9]*128, obs[10]*128, obs[11]*128, obs[12]*128

    plt.show()

    # Evaluate the policy
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    # print(f"Evaluation: mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    def print_best_and_worst():
        if None not in max_stats:
            print(f"{max_damage} of {global_max_damage}, that's {max_damage/global_max_damage*100:.2f}%")
            print(f"Fireball used: {max_stats[0]}")
            print(f"Frostbolt used: {max_stats[1]}")
            print(f"BloodMoonCrescent used: {max_stats[2]}")
            print(f"Blaze used: {max_stats[3]}")
            print(f"ScorchDot used: {max_stats[4]}")
            print(f"Combustion used: {max_stats[5]}")
        else:
            print("fuck you")

            # print("------"*5)
            # print("------"*5)
            # print worst one
            """
        if None not in min_stats:
            print(min_damage, "of", global_max_damage)
            print(f"Fireball used: {min_stats[0]}")
            print(f"Frostbolt used: {min_stats[1]}")
            print(f"BloodMoonCrescent used: {min_stats[2]}")
            print(f"Blaze used: {min_stats[3]}")
            print(f"ScorchDot used: {min_stats[4]}")
            print(f"Combustion used: {min_stats[5]}")
        else:
            print("fuck you")
            """
    print_best_and_worst()

    env.close()


global_max_damage = 4242.5


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
            max_damage = global_max_damage
            max_count = 128
            last_spell = 0
            num_features = 1 + 2 * len(self.env.spells) + 1  # total_damage + cd*6 (foreach spell) + last_spell
            self.observation_space = spaces.Box(
                low=np.zeros(num_features, dtype=np.float32),  # All lows are 0
                high=np.array([max_damage] + ([max_count]+[max_cooldown])*len(self.env.spells) + [last_spell]),
                # low=np.float32(0),
                # high=np.float32(np.inf),
                # shape=(num_features,),
                dtype=np.float32
            )

        def step(self, action):
            # action is a np.int64

            action_name = list(self.env.spells.keys())[action]
            self.env.step(action_name)
            obs = self.env.get_results()
            reward = obs[0]  # Total-damage in numpy get_results()
            done = self.tick_count >= 128
            self.tick_count += 1  # Increment tick count for Done
            return obs, reward, done, {}

        def reset(self):
            self.tick_count = 0  # Reset tick count
            return self.env.reset()

        def render(self, mode='human'):
            self.env.render()

    def initialize_population(pop_size, action_space, sequence_length):
        return [np.random.randint(action_space.n, size=sequence_length)
                for _ in range(pop_size)]

    def evaluate_population(env, population):
        fitness_scores = []
        for solution in population:
            fitness = evaluate_solution(env, solution)[0]
            fitness_scores.append((solution, fitness))
        return fitness_scores

    def evaluate_solution(env, solution):
        obs = env.reset()
        total_reward = 0
        best_reward = 0
        for action in solution:
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if reward > best_reward:
                best_reward = reward
            if done:
                break
        return [total_reward, best_reward]

    def select_top_solutions(population, fitnesses, top_k=0.1):
        sorted_indices = np.argsort(fitnesses)[::-1]  # Sort fitnesses in descending order
        top_cutoff = int(len(population) * top_k)
        top_indices = sorted_indices[:top_cutoff]
        return [population[i] for i in top_indices]

    def crossover(parent1, parent2):
        # Single-point crossover
        crossover_point = np.random.randint(len(parent1))
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child

    def mutate(solution, mutation_rate, action_space):
        for i in range(len(solution)):
            if np.random.rand() < mutation_rate:
                solution[i] = np.random.randint(action_space.n)
        return solution

    def genetic_algorithm(env, pop_size, generations, sequence_length, mutation_rate=0.01):
        # Initialize population
        population = initialize_population(pop_size, env.action_space, sequence_length)

        for generation in range(generations):
            # Evaluate all solutions in the population
            fitnesses = [evaluate_solution(env, sol)[0] for sol in population]
            best_damage = max([evaluate_solution(env, sol)[1] for sol in population])

            # Select the top-performing solutions based on their fitness
            # This could be a function to sort the fitnesses and select the top indices
            top_indices = np.argsort(fitnesses)[-int(0.1 * len(fitnesses)):]  # Get top 10% indices

            # Using indices to select from the population
            top_solutions = [population[i] for i in top_indices]
            print(f"Generation {generation+1}: Max Damage {best_damage} of {global_max_damage}, that's {best_damage/global_max_damage}%")

            # Randomly choose two unique indices from the list of top solution indices
            selected_indices = np.random.choice(top_indices, 2, replace=False)
            parent1, parent2 = population[selected_indices[0]], population[selected_indices[1]]

            # Create the next generation
            new_population = []
            while len(new_population) < pop_size:
                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate, env.action_space)
                new_population.append(child)

            population = new_population

            # Logging the progress
            # print(f"Generation {generation+1}: Max Damage {(fitnesses[0])}")
            # print(f"Generation {generation+1}: Max Damage {(top_solutions[0])}")

        return population

    # Initialize the custom environment
    env = DummyVecEnv([lambda: RunSimEnv()])
    # env = RunSimEnv()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # This is specific to certain OS environments

    # Set up plot for live updating
    fig, ax = plt.subplots()

    # Initialize & train the RL model
    model = PPO("MlpPolicy", env, verbose=1,
                gamma=0.9,
                n_steps=2048,
                ent_coef=0.01,
                batch_size=2048,
                gae_lambda=0.95)\
                .learn(total_timesteps=10000)
    
    # Save the model
    save_dir = "/tmp/gym/"
    os.makedirs(save_dir, exist_ok=True)
    model.save(f"{save_dir}/ppo_runsim")

    # Load model
    """
    # Optionally, you can reload it
        # model = PPO.load(f"{save_dir}/ppo_runsim", env=env, verbose=1)
    # show the save hyperparameters
        # print(f"loaded: gamma={model.gamma}, n_steps={model.n_steps}")
    # as the environment is not serializable, we need to set a new instance of the environment
        # model.set_env(DummyVecEnv([lambda: RunSimEnv()]))
        # model.learn(8000)
    """

    env = RunSimEnv()
    obs = env.reset()
    draw_plot(env, model, ax, obs)

    pop_size = 100
    generations = 600
    sequence_length = 128

    best_population = genetic_algorithm(env, pop_size, generations, sequence_length)
    fitness_scores = evaluate_population(env, best_population)
    best_solution, best_fitness = max(fitness_scores, key=lambda x: x[1])

    print("Best Performing Solution:")
    print("Sequence of Actions (Spells):", best_solution)
    print("Total Fitness (e.g., Total Damage):", best_fitness, f"of {global_max_damage}")


if __name__ == "__main__":
    run_simulation()

# TODO: Ab wann ist overfitting
# TODO: Parameter durch Cross-Entropy versuchen
# TODO: Datenbank auslagern?
# TODO: Mal bissl aufräumen...
# TODO: Hat Push geklappt?

"""
6 Stages of Success:
    1. Stop casting spells, that are on cooldown (edit: Klappt zu 90%)
    2. Stop casting Frostbolt, because its useless (edit: YES ES KLAPPT)
    3. Stop casting DoT, when its already active
    4. Casting Combustion and BloodMoon on Cooldown, because it does most damage
    5. Alternating between Fireball and Blaze
    6. Waiting for BloodMoon cooldown when buff-cd is going to expire soon
"""
