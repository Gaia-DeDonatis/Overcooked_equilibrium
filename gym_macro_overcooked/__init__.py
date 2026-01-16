from gym.envs.registration import register

register(
    id='Overcooked-v0',
    entry_point='gym_macro_overcooked.overcooked:Overcooked',
)

# mapA
register(
    id='Overcooked-MA-v0',
    entry_point='gym_macro_overcooked.overcooked_MA:Overcooked_MA',
)

# mapA_ban
register(
    id='Overcooked-MA-v3',
    entry_point='gym_macro_overcooked.overcooked_MA_ban_get_plate:Overcooked_MA_ban_get_plate',
)

# mapB, mapB_lowuncertainty
register(
    id='Overcooked-MA-v1',
    entry_point='gym_macro_overcooked.overcooked_MA_mapB:Overcooked_MA_mapB',
)

# mapB_ban, mapB_lowuncertainty_ban
register(
    id='Overcooked-MA-v4',
    entry_point='gym_macro_overcooked.overcooked_MA_mapB_ban_get_plate:Overcooked_MA_mapB_ban_get_plate',
)

# mapA_lowuncertainty
register(
    id='Overcooked-MA-v2',
    entry_point='gym_macro_overcooked.overcooked_MA_lowuncertainty:Overcooked_MA_lowuncertainty',
)

# mapA_lowuncertainty_ban
register(
    id='Overcooked-MA-v5',
    entry_point='gym_macro_overcooked.overcooked_MA_lowuncertainty_ban_get_plate:Overcooked_MA_lowuncertainty_ban_get_plate',
)