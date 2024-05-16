from data.records import *

spell_map = {
    0: 'Fireball',
    1: 'Frostbolt',
    2: 'BloodMoonCrescent',
    3: 'Blaze',
    4: 'ScorchDot',
    5: 'Combustion'
}


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
            run_sim.character.deal_damage(self.damage)
            return True
        return False


class Frostbolt(Spell):
    def cast(self, run_sim):
        if super().cast(run_sim):
            run_sim.character.deal_damage(self.damage)
            return True
        return False


class BloodMoonCrescent(Spell):

    def cast(self, run_sim):
        if super().cast(run_sim):
            run_sim.character.deal_damage(self.damage)
            return True
        return False


class Blaze(Spell):
    def cast(self, run_sim):
        if super().cast(run_sim):
            character = run_sim.character
            if character.last_spell == "Fireball":
                applied_damage = self.damage * 5
                run_sim.character.deal_damage(applied_damage)
                # print(f"Blaze cast after Fireball, damage applied: {applied_damage}")
            else:
                run_sim.character.deal_damage(self.damage)
                # print(f"Blaze cast without Fireball, damage applied: {self.damage}")
            return True
        return False


class ScorchDot(Spell):
    def __init__(self, name, cooldown, duration, damage):
        super().__init__(name, cooldown, damage)
        self.duration = duration

    def cast(self, run_sim):
        if super().cast(run_sim):
            run_sim.character.inflict_dot(self.damage, self.duration)
            return True
        return False


class Combustion(Spell):
    def __init__(self, name, cooldown, duration, damage_increase):
        super().__init__(name, cooldown, duration)
        self.duration = duration
        self.damage_increase = damage_increase

    def cast(self, run_sim):
        if super().cast(run_sim):
            run_sim.character.activate_buff(self.name, self.duration, self.damage_increase)
            return True
        return False


class TrainingDummy:

    def __init__(self):
        self.debuff_damage_increase = 0.0
        self.debuff_name = ""
        self.debuff_timer = 0
        self.debuff_active = False

        self.dot_damage = 0
        self.dot_timer = 0

        self.damage_taken = 0

    def apply_dot(self, dot_damage, duration):
        self.dot_timer = duration
        self.dot_damage = dot_damage

    def tick(self):
        # Check Dot-Timer
        if self.dot_timer > 0:
            self.dot_timer -= 1

    def activate_debuff(self, name, duration, damage_increase):
        self.debuff_active = True
        self.debuff_timer = duration
        self.debuff_name = name
        self.debuff_damage_increase = damage_increase

    def calc_damage(self, damage):
        if self.debuff_active:
            self.damage_taken += damage * (1 + self.debuff_damage_increase)
        else:
            self.damage_taken += damage


class Character:
    def __init__(self, trainingsdummy):
        self.buff_damage_increase = 0.0
        self.buff_name = ""
        self.buff_active = False
        self.buff_timer = 0
        self.last_spell = None
        self.damage_done = 0
        self.trainingsdummy = trainingsdummy

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
        if self.trainingsdummy.dot_timer > 0:
            self.deal_damage(self.trainingsdummy.dot_damage)

    def deal_damage(self, damage):
        if self.buff_active:
            self.trainingsdummy.calc_damage(damage * (1 + self.buff_damage_increase))
        else:
            self.trainingsdummy.calc_damage(damage)

    def inflict_dot(self, dot_damage, duration):
        self.trainingsdummy.apply_dot(dot_damage, duration)


class RunSim:
    def __init__(self):
        self.training_dummy = TrainingDummy()
        self.character = Character(self.training_dummy)

        # List of spells
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
        self.character.tick()
        self.training_dummy.tick()

        # Check Cooldown Tick
        for spell in self.spells.values():
            spell.cooldown_tick()

        # Create output for visualisation
        self.render()

    def render(self):

        print(f"Total Damage: {self.training_dummy.damage_taken} damage, that's {self.training_dummy.damage_taken/4242.5*100:.2f}%")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"Remaining DoT Duration: {self.training_dummy.dot_timer}")
        print(f"Remaining Buff Duration: {self.character.buff_timer}."
              f" CD-Buff: {self.spells['Combustion'].current_cooldown}."
              f" CD-Moon: {self.spells['BloodMoonCrescent'].current_cooldown}")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"Spell cast last time: {self.character.last_spell}")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")

    def simulate(self, ticks_amount):

        # Simulate a copied_list:
        copied_list = record16052024_3  # See file records.py
        better_list = copied_list.replace('[', '').replace(']', '').replace('.', '').split(' ')
        for i in range(ticks_amount):  # Simulate Ticks/Seconds
            # chosen_spell = spell_map[int(better_list[i])]
            # self.step(chosen_spell)

            # Simulator for playing yourself:

            print(f"This is input {i} of {ticks_amount}. Enter a spell")
            chosen_input = int(input())
            if chosen_input in spell_map:
                chosen_spell = spell_map[chosen_input]
                self.render()
                self.step(chosen_spell)
            else:
                print("Invalid input, please enter a valid spell number.")
                continue  # Skip to the next iteration if the input is invalid

        print("Total damage taken:", self.training_dummy.damage_taken)


if __name__ == "__main__":
    env = RunSim()
    env.simulate(128)
    # Self simulated max damage = 4250 in 128 ticks, e.g. global_max_damage
