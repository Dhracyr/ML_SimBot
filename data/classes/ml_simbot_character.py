from data.classes.ml_simbot_buffs import Buff


class Character:
    def __init__(self, trainingsdummy):
        self.buffs = []
        self.last_spell = None
        self.damage_done = 0
        self.trainingsdummy = trainingsdummy

    def activate_buff(self, name, duration, damage_increase, stacks):
        for buff in self.buffs:
            if buff.name == name:
                buff.timer = duration
                return
        # Add new buff
        new_buff = Buff(name, duration, damage_increase, stacks)
        self.buffs.append(new_buff)

    def is_buff_active(self, name):
        # Check if a specific buff is active
        for buff in self.buffs:
            if buff.name == name:
                return True
        return False

    def increment_buff_stacks(self, name, amount):
        for buff in self.buffs:
            if buff.name == name:
                buff.stacks += amount
                return buff.stacks

    def reset_buff_to_duration(self, name, duration):
        for buff in self.buffs:
            if buff.name == name:
                buff.timer = duration

    def get_stacks_of_buff(self, name):
        for buff in self.buffs:
            if buff.name == name:
                return buff.stacks

    # Check Buff-Timer
    def tick(self):
        self.buffs = [buff for buff in self.buffs if buff.tick()]
        if self.trainingsdummy.dot_timer > 0:
            self.deal_damage(self.trainingsdummy.dot_damage)

    def deal_damage(self, damage):
        total_damage_increase = sum(buff.damage_increase for buff in self.buffs)
        self.trainingsdummy.calc_damage(damage * (1 + total_damage_increase))

    def inflict_dot(self, dot_damage, duration):
        self.trainingsdummy.apply_dot(dot_damage, duration)
