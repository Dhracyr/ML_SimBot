spell_map = {
    0: 'Fireball',
    1: 'Frostbolt',
    2: 'BloodMoonCrescent',
    3: 'Blaze',
    4: 'ScorchDot',
    5: 'Combustion',
    6: 'LivingFlame'
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
    def __init__(self, name, cooldown, duration, damage, stacks):
        super().__init__(name, cooldown, damage)
        self.stacks = stacks
        self.duration = duration

    def cast(self, run_sim):
        if super().cast(run_sim):
            run_sim.character.deal_damage(self.damage)
            if run_sim.character.is_buff_active("LivingFlame"):
                run_sim.character.activate_buff(self.name, self.duration, self.damage, self.stacks)
            else:
                run_sim.character.increment_buff_stacks("LivingFlame", 1)
                run_sim.character.reset_buff_to_duration("LivingFlame", self.duration)

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
    def __init__(self, name, cooldown, duration, damage, damage_increase):
        super().__init__(name, cooldown, damage)
        self.duration = duration
        self.damage_increase = damage_increase

    def cast(self, run_sim):
        if super().cast(run_sim):
            run_sim.character.activate_buff(self.name, self.duration, self.damage_increase)
            return True
        return False


class LivingFlame(Spell):
    def __init__(self, name, cooldown, duration, damage):
        super().__init__(name, cooldown, damage)
        self.duration = duration

    def cast(self, run_sim):
        if super().cast(run_sim):
            if run_sim.character.is_buff_active("LivingFlame"):
                live_stacks = run_sim.character.increment_buff_stacks("LivingFlame", 0)
                applied_damage = self.damage + 10 * live_stacks
                run_sim.character.deal_damage(applied_damage)
                print("LivingFlame was casted with", live_stacks, "stacks!")
            else:
                run_sim.character.deal_damage(self.damage)
                print("LivingFlame was casted without stacks :(")
