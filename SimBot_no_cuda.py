import os
import gym
import numpy as np

from ml_simbot_spells import *
from ml_simbot_trainingsdummy import *
from ml_simbot_character import *

from gym import spaces
from matplotlib import pyplot as plt


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
        cast_counts = [self.spell_cast_count[name] / GLOBAL_MAX_TICKS for name in self.spells.keys()]
        last_spell = [1] if self.character.last_spell == 'Blaze' else [0]
        state = [self.training_dummy.damage_taken] + cooldowns + cast_counts + last_spell
        return np.array(state, dtype=np.float32)


def run_simulation():
    class RunSim:
        def __init__(self):
            self.character = Character()
            self.training_dummy = TrainingDummy(self.character)  # Because the trainingsdummy must know,
            # List of spells                                            # the buff of the player
            self.spells = {

                'Fireball': Fireball('Fireball', 0, 10),  # Name, Cooldown, Damage
                'Frostbolt': Frostbolt('Frostbolt', 0, 3),  # Name, Cooldown, Damage
                'BloodMoonCrescent': BloodMoonCrescent('BloodMoonCrescent', 10, 80),  # Name, Cooldown, Damage
                'Blaze': Blaze('Blaze', 0, 5),  # Name, Cooldown, Damage
                'ScorchDot': ScorchDot('ScorchDot', 0, 20, 5),  # Name, Cooldown, Duration, Damage
                'Combustion': Combustion('Combustion', 60, 25, 0.5)  # Name, Cooldown, Duration, Damage_increase
            }
            self.spell_cast_count = {name: 0 for name in self.spells.keys()}

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

        """
        def reset(self):
            self.character = Character()
            self.training_dummy = TrainingDummy(self.character)
            for spell in self.spells.values():
                spell.current_cooldown = 0
            self.spell_cast_count = {name: 0 for name in self.spells.keys()}
            return self.get_results()
        """

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

        def get_action_space_n(self):
            return len(self.spells)

        def get_results(self):
            cooldowns = [spell.current_cooldown / 60 for _, spell in self.spells.items()]
            cast_counts = [self.spell_cast_count[name] / GLOBAL_MAX_TICKS for name in self.spells.keys()]
            last_spell = [1] if self.character.last_spell == 'Blaze' else [0]
            state = [self.training_dummy.damage_taken] + cooldowns + cast_counts + last_spell
            return np.array(state, dtype=np.float32)

    def initialize_population(pop_size, action_space, sequence_length):
        return [np.random.randint(action_space, size=sequence_length)
                for _ in range(pop_size)]

    def evaluate_population(ep_env, population):
        fitness_scores = []
        for solution in population:
            fitness = evaluate_solution(ep_env, solution)[0]
            fitness_scores.append((solution, fitness))
        return fitness_scores

    def evaluate_solution(es_env, solution):
        spell_map = {
            0: 'Fireball',
            1: 'Frostbolt',
            2: 'BloodMoonCrescent',
            3: 'Blaze',
            4: 'ScorchDot',
            5: 'Combustion'
        }
        for name_action in solution:
            chosen_spell = spell_map[name_action]
            es_env.step(chosen_spell)

        damage_done_with_solution = es_env.training_dummy.damage_taken

        return damage_done_with_solution

    def crossover(parent1, parent2):
        # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def tournament_selection(ts_env, population, k=3):
        idxes = [np.random.randint(len(population)) for _ in range(k)]
        selected = [population[idx] for idx in idxes]
        selected_fitness = [evaluate_solution(ts_env, sol) for sol in selected]

        # Find the index of the solution with the highest total_reward manually
        max_fitness = -float('inf')  # Assumes fitness can't be lower than this
        best_index = 0
        for i, fitness in enumerate(selected_fitness):
            if fitness > max_fitness:  # Assuming the first element is total_reward
                max_fitness = fitness
                best_index = i

        return selected[best_index]

    def mutate(solution, mutation_rate, action_space):
        for i in range(len(solution)):
            if np.random.rand() < mutation_rate:
                solution[i] = np.random.randint(action_space)
        return solution

    def adapt_mutation_rate(current_rate, generations_without_improvement, max_rate=0.05, min_rate=0.01):
        if generations_without_improvement > 10:  # No improvement for 10 generations
            new_rate = min(current_rate * 1.3, max_rate)
            print("No improvement since", generations_without_improvement, "mutation_rate is getting raised to", new_rate)
        else:
            new_rate = max(current_rate * 0.95, min_rate)
            print("Improvement since", generations_without_improvement, "mutation_rate is getting lowered to", new_rate)
        return new_rate

    def draw_plot_all_gen(line, ax, fig, list_best_damage, list_generation):

        ax.set_xlabel('Generation')
        ax.set_ylabel('Total Damage')
        ax.set_xlim(0, global_generations)  # Use the number of generations for the x-axis limit
        ax.set_ylim(0, GLOBAL_MAX_DAMAGE*1.1)  # Example: set maximum possible damage

        line.set_data(list_generation, list_best_damage)

        fig.canvas.draw()
        fig.canvas.flush_events()

        # Animated Version
        """
        line.set_data(list_generation, list_best_damage)
        ax.relim()
        ax.autoscale_view()

        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()
        """
    list_all_solutions = []

    def genetic_algorithm(ga_env, pop_size, generations, sequence_length, mutation_rate):
        action_space_n = ga_env.get_action_space_n()
        # Initialize population
        population = initialize_population(pop_size, action_space_n, sequence_length)
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # This is specific to certain OS environments
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        list_best_damages = []
        list_generations = []

        generations_without_improvement = 0
        saved_damage_peak = 0.0

        ax.axhline(y=GLOBAL_CURRENT_RECORD, color='g', linestyle='--', linewidth=1, label='Current record')
        ax.axhline(y=GLOBAL_MAX_DAMAGE, color='r', linestyle='-', linewidth=1, label='Max Damage')

        line1, = ax.plot(list_generations, list_best_damages, linestyle='-', color='b')

        for generation in range(generations):

            # Evaluate all solutions in the population via RunSim()
            all_damage_as_list = [evaluate_solution(ga_env, sol) for sol in population]
            all_damage_as_list = np.array(all_damage_as_list, dtype=np.float32)
            best_damage = np.max(all_damage_as_list)

            # Alter mutation_rate
            if saved_damage_peak == 0:
                saved_damage_peak = best_damage
            if best_damage == saved_damage_peak:
                generations_without_improvement += 1
            else:
                generations_without_improvement = 0
                saved_damage_peak = best_damage

            # Adapt mutation_rate if there are too many generations without improvement
            mutation_rate = adapt_mutation_rate(mutation_rate, generations_without_improvement, global_max_mutation_rate, global_min_mutation_rate)

            # Print the damage that the generation did
            print(f"Generation {generation}: Max Damage {best_damage} of {GLOBAL_MAX_DAMAGE}, that's {best_damage/GLOBAL_MAX_DAMAGE*100:.2f}%")

            # Set up plot for live updating
            list_best_damages.append(best_damage)
            list_generations.append(generation)

            # Draw live-plot
            draw_plot_all_gen(line1, ax, fig, list_best_damages, list_generations)
            plt.pause(0.01)

            # Save generation in list
            list_all_solutions.append(population)

            # Create the next generation
            new_population = []
            while len(new_population) < pop_size:
                # Random of top 10%
                # index1, index2 = np.random.choice(len(top_indices), 2, replace=False)
                # parent1, parent2 = population[index1], population[index2]

                # Tournament Selection
                parent1 = tournament_selection(ga_env, population, global_tournament_k_amount)
                parent2 = tournament_selection(ga_env, population, global_tournament_k_amount)

                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, mutation_rate, action_space_n)
                child2 = mutate(child2, mutation_rate, action_space_n)

                new_population.extend([child1, child2])

            population = new_population

        plt.ioff()
        plt.show()
        return population

    env = RunSim()

    pop_size = global_pop_size
    generations = global_generations
    sequence_length = GLOBAL_MAX_TICKS

    best_population = genetic_algorithm(env, pop_size, generations, sequence_length, start_population_mutation_rate)
    fitness_scores = evaluate_population(env, best_population)
    best_solution, best_fitness = max(fitness_scores, key=lambda x: x[1])

    print("Best Performing Solution:")
    print("Sequence of recent actions (spells):", best_solution)
    print("Sequence of best actions (spells):", max(list_all_solutions, key=lambda x: x[1]))
    # print(list_all_solutions)


# const stats
GLOBAL_MAX_DAMAGE = 4242.5
GLOBAL_CURRENT_RECORD = 4112.5
GLOBAL_MAX_TICKS = 128

# duration
global_generations = 100

# parameter for cross-entropy
global_pop_size = 30  # 50
global_population_top_n_index = 0.1
start_population_mutation_rate = 0.01  # 0.01
global_tournament_k_amount = 3  # 10% of pop_size probably
global_max_mutation_rate = 0.015  # 0.015
global_min_mutation_rate = 0.005  # 0.005

# TODO: Cross-over-rate?
# TODO: Reward Function that punishes similarity
# TODO: Diversity Checks
# TODO: New Spell: Fireball gives a stack of "flaming", each stack increases the damage of the new spell by 15%, stackable for 20 Stacks


if __name__ == "__main__":
    run_simulation()


# TODO: Parameteroptimierung, da Overfittung bei 95%
# TODO: Plotten von Generationen in Farben

# TODO: Parameter durch Cross-Entropy versuchen
# TODO: Datenbank auslagern?

"""
6 Stages of Success:
    1. Done!        Stop casting spells, that are on cooldown
    2. Done!        Stop casting Frostbolt, because its useless
    3. Done!        Stop casting DoT, when its already active
    4. Done!        Casting Combustion and BloodMoon on Cooldown, because it does most damage
    5. Done!        Alternating between Fireball and Blaze
    6. Oftentimes!  Waiting for BloodMoon cooldown when buff-cd is going to expire soon
"""