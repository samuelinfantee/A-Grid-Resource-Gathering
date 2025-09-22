# A-Grid-Resource-Gathering
This repo implements an A* search agent that collects the required resources on a 5×5 grid while respecting backpack capacity, then returns to base

The python code sinfante_Intro AI_Project 1 part 2 (code).py contains the A* algorithm solution with 2 admissible heuristics (manhattan distance and zero heuristic), and with 4 test maps.

# install the only external dependency
pip install matplotlib

# Editing maps & parameters

At the top:

Resources (fixed for all tests unless you change them):
resources = {
    "stones":   [(1,3), (3,0), (4,2)],
    "irons":    [(2,1), (4,4)],
    "crystals": [(0,4)]
}

Terrain costs:
terrain_types = {"grassland": 1, "hills": 2, "swamp": 3, "mountain": 4}

Backpack capacity:
backpack_capacity = 2

Goals:
goal = {"stones": 3, "irons": 2, "crystals": 1}

Test maps (TEST_MAPS): four 5×5 boards; tweak or add your own:
TEST_MAPS = {}



