from gym.envs.registration import register


register(
    id='Overcooked-equilibrium-v0',
    entry_point='gym_macro_overcooked.overcooked_equilibrium:Overcooked_equilibrium',
)

register(
    id='Overcooked-MA-equilibrium-v0',
    entry_point='gym_macro_overcooked.overcooked_MA_equilibrium:Overcooked_MA_equilibrium',
)
