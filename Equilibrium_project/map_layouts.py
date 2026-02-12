# map_layouts.py

# --- REFERENCE LEGEND (From overcooked_equilibrium.py) ---
# 0 = Space
# 1 = Counter
# 2 = Agent (1 or 2)
# 4 = Lettuce
# 5 = Plate
# 6 = Knife (Chopping Board)
# 7 = Delivery (Serving Station)

def get_dim(grid):
    return [len(grid), len(grid[0])]

# Cramped Room → low-level coordination challenges: in this shared, confined space it is very easy for the agents to collide.

GRID_CRAMPED = [
    [1, 1, 6, 1, 1],
    [4, 0, 0, 2, 4],
    [1, 2, 0, 0, 1],
    [1, 5, 1, 7, 1]
]

# Asymmetric Advantages → tests whether players can choose high-level strategies that play to their strengths.
GRID_ASYMMETRIC = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [4, 0, 1, 7, 1, 4, 1, 0, 7],
    [1, 0, 0, 0, 6, 0, 0, 0, 1],
    [1, 2, 0, 0, 6, 0, 0, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Coordination Ring → players must coordinate to travel between the bottom left and top right corners of the layout.
GRID_RING = [
    [1, 1, 1, 6, 1],
    [1, 0, 0, 2, 1],
    [5, 0, 1, 0, 1],
    [4, 2, 0, 0, 1],
    [1, 4, 7, 1, 1]
]

# Forced Coordination → removes collision coordination problems, and forces players to develop a high-level joint strategy, since neither player can serve a dish by themselves.
GRID_FORCED = [
    [1, 1, 1, 6, 1],
    [4, 0, 1, 2, 6],
    [4, 2, 1, 0, 1],
    [5, 0, 1, 0, 1],
    [1, 1, 1, 7, 1]
]

# Counter Circuit → involves a non-obvious coordination strategy, where onions are passed over the counter to the pot, rather than being carried around
GRID_CIRCUIT = [
    [1, 1, 1, 6, 6, 1, 1, 1],
    [1, 0, 2, 0, 0, 0, 0, 1],
    [5, 0, 1, 1, 1, 1, 1, 7],
    [1, 0, 0, 0, 0, 0, 2, 1],
    [1, 1, 1, 4, 4, 1, 1, 1]
]


LAYOUT_REGISTRY = {
    "cramped":    {"grid": GRID_CRAMPED,    "dim": get_dim(GRID_CRAMPED)},
    "asymmetric": {"grid": GRID_ASYMMETRIC, "dim": get_dim(GRID_ASYMMETRIC)},
    "ring":       {"grid": GRID_RING,       "dim": get_dim(GRID_RING)},
    "forced":     {"grid": GRID_FORCED,     "dim": get_dim(GRID_FORCED)},
    "circuit":    {"grid": GRID_CIRCUIT,    "dim": get_dim(GRID_CIRCUIT)},
}