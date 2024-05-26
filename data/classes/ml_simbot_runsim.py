import numpy as np

from data.classes.ml_simbot_character import *
from data.classes.ml_simbot_trainingsdummy import *
from data.classes.ml_simbot_spells import *
from data.global_variables import GLOBAL_MAX_TICKS


class RunSim:
    def __init__(self):
        self.training_dummy = TrainingDummy()
        self.character = Character(self.training_dummy)

        # List of spells
        self.spells = {

            'Fireball': Fireball('Fireball', 0, 5, 10, 1),  # Name, Cooldown, Duration, Damage
            'Frostbolt': Frostbolt('Frostbolt', 0, 3),  # Name, Cooldown, Damage
            'BloodMoonCrescent': BloodMoonCrescent('BloodMoonCrescent', 10, 80),  # Name, Cooldown, Damage
            'Blaze': Blaze('Blaze', 0, 5),  # Name, Cooldown, Damage
            'ScorchDot': ScorchDot('ScorchDot', 0, 20, 5),  # Name, Cooldown, Duration, Damage
            'Combustion': Combustion('Combustion', 60, 25, 0, 0.5, 1),  # Name, Cooldown, Duration, Damage, Damage_increase
            'LivingFlame': LivingFlame('LivingFlame', 0, 5, 50)  # Name, Cooldown, Duration, Damage
        }
        self.spell_cast_count = {name: 0 for name in self.spells.keys()}

    def step(self, action_name):
        # Check last_casted_spell & increase cast_count
        spell = self.spells[action_name]
        if spell.cast(self):
            self.character.last_spell = spell.name
            self.spell_cast_count[action_name] += 1
            if action_name == "LivingFlame":
                print("Here!")

        # Check Dot-Spell/Buff Tick
        self.character.tick()
        self.training_dummy.tick()

        # Check Cooldown Tick
        for spell in self.spells.values():
            spell.cooldown_tick()

        # Create output for visualisation
        # self.render()

    def reset(self):
        self.training_dummy = TrainingDummy()
        self.character = Character(self.training_dummy)
        for spell in self.spells.values():
            spell.current_cooldown = 0
        self.spell_cast_count = {name: 0 for name in self.spells.keys()}
        # return self.get_results()

    def render(self):
        """
        print(f"Total Damage: {self.training_dummy.damage_taken} damage")
        print(f"- - - - - - - - - - - - - - - - - - - - - - - - - -")
        print(f"Remaining DoT Duration: {self.training_dummy.dot_timer}")
        print(f"Remaining Buff Duration: {self.character.buff_timer}). "
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
    """
    def get_results(self):
        cooldowns = [spell.current_cooldown / 60 for _, spell in self.spells.items()]
        cast_counts = [self.spell_cast_count[name] / GLOBAL_MAX_TICKS for name in self.spells.keys()]
        last_spell = [1] if self.character.last_spell == 'Blaze' else [0]
        state = [self.training_dummy.damage_taken] + cooldowns + cast_counts + last_spell
        return np.array(state, dtype=np.float32)
    """