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
