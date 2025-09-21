import math
from dataclasses import dataclass
from typing import Tuple, FrozenSet
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#Intro to AI - Project 1 part 2
#By: Samuel Infante Trillos, U42773730

#Resource loc_agents.
resources = {
    "stones": [(0,3), (2,1), (2,2)],
    "irons": [(2,4), (3,3)],
    "crystals": [(4,4)]
}



# 5x5 terrain grids: 
TEST_MAPS = {
    "Map1_flat": [
        ['grassland','grassland','grassland','grassland','grassland'],
        ['grassland','grassland','grassland','grassland','grassland'],
        ['grassland','grassland','hills',     'grassland','grassland'],
        ['grassland','grassland','grassland','grassland','grassland'],
        ['grassland','grassland','grassland','grassland','grassland'],
    ],
    "Map2_hills_swamp": [
        ['grassland','hills','hills','grassland','grassland'],
        ['grassland','hills','swamp','swamp','grassland'],
        ['hills','hills','hills','swamp','hills'],
        ['grassland','grassland','hills','grassland','grassland'],
        ['grassland','hills','grassland','hills','grassland'],
    ],
    "Map3_blocker": [
        ['grassland','grassland','grassland','grassland','hills'],
        ['grassland','mountain','mountain','mountain','grassland'],
        ['grassland','hills','hills','hills','grassland'],
        ['grassland','mountain','mountain','mountain','grassland'],
        ['hills','grassland','hills','grassland','hills'],
    ],
    "Map4_hardcore": [
        ['grassland','mountain','grassland','mountain','hills'],
        ['mountain','grassland','mountain','mountain','grassland'],
        ['mountain','hills','grassland','mountain','swamp'],
        ['swamp','swamp','mountain','swamp','mountain'],
        ['hills','grassland','hills','swamp','swamp'],
    ],
}



#Terrain types and their movement costs.
terrain_types = {"grassland": 1, "hills": 2, "swamp": 3, "mountain": 4}
#Goal state for the resources to be collected.
goal = {"stones": 3, "irons": 2, "crystals": 1}
#Maximum capacity of resources in the backpack at any time.
backpack_capacity = 2
grid_x, grid_y = 5, 5  #Dimensions of the grid.


#--------------------------------------------------------------------------------------------------------
#Dataset
#--------------------------------------------------------------------------------------------------------
# Fixed order for the three resource types.
resource_order = ('stones', 'irons', 'crystals')

# Quick index lookup: 'stones' -> 0, 'irons' -> 1, 'crystals' -> 2
resource_id = {r: i for i, r in enumerate(resource_order)}

def counts_to_tuple(d):
    # Build a list of counts in the specified resource_order.
    result = []
    for r in resource_order:
        result.append(d[r]) 
    # Convert the list to a tuple
    return tuple(result)

#Check if we’ve returned_base enough of each resource to meet the goal.
def goal_met(deliv_tuple):  
    for i in range(3):
        res_name = resource_order[i]
        required_count = goal[res_name]
        delivered_count = deliv_tuple[i]
        # If any delivered amount is below its goal, we haven't met the goal.
        if delivered_count < required_count:
            return False
    return True

@dataclass(frozen=True)
class State:
    # Agent position on the grid
    loc_agent: Tuple[int, int] # (x, y)
    # What the agent is currently carrying (in backpack), as counts by type
    contents: Tuple[int, int, int]
    # Total amounts already deposited at base (also in the same order)
    returned_base: Tuple[int, int, int]
    # Resources still available on the map to pick up.
    remaining: FrozenSet[Tuple[int, int, str]]

def make_start_state(): #Initiate the start state of the agent at the base with an empty backpack and all resources available.
    rem = []
    for rname, locs in resources.items(): #Automatically gather all resources and loc_agents from the resources dictionary.
        for (x, y) in locs:
            rem.append((x, y, rname)) #Add each resource loc_agent and type to the remaining resources list.

    return State(
        loc_agent=(0, 0),               # start at base
        contents=(0, 0, 0),       # backpack empty
        returned_base=(0, 0, 0),      # nothing returned_base yet
        remaining=frozenset(rem)  # all resource tiles are initially available
    )

#--------------------------------------------------------------------------------------------------------
#Helper functions
#--------------------------------------------------------------------------------------------------------

#1)
def get_successors(loc_agent): #We use this function to get all valid successor positions from the current position in the grid.
    directions = [(0,1), (1,0), (0, -1), (-1,0)]  #Right, Down, Left, Up
    successors = []
    for d in directions:
        new_x, new_y = loc_agent[0] + d[0], loc_agent[1] + d[1]
        if 0 <= new_x < grid_x and 0 <= new_y < grid_y: #Each successor must be within the grid boundaries.
            successors.append((new_x, new_y)) #We add the successor if its valid.
    return successors

#2)
def step_cost(a, b): #This function calculates the cost to move from tile a to tile b. The cost is determined by the terrain type of tile b. We check this cost in the terrain_grid variable.
    tx, ty = b
    tname = terrain_grid[tx][ty] 
    return terrain_types[tname]

#3)
#This function basically checks for the next state if we should deposit resources (if we are at the base) or pick up resources (if we are at a resource tile and have capacity).
def step(current: State, successor: Tuple[int,int]) -> Tuple[State, float]:
    #In this function we move into successor from current. We apply the auto deposit if we are at the base (0,0), and autopickup if capacity allows it.
    move_cost = step_cost(current.loc_agent, successor)
    #We copy the state from the current node.
    remaining_res = set(current.remaining)
    returned_base_res = list(current.returned_base)
    inventory = list(current.contents)

    #Option 1) We check if the current tile is the base (0,0) and we deposit all resources in the backpack.
    if successor == (0,0) and sum(inventory) > 0:
        for i in range(3): #For each resource type.
            returned_base_res[i] += inventory[i] #Deposit all resources of that type.
            inventory[i] = 0 #Empty the backpack for that resource type.


    #Option 2) We check if we have a resource at current tile and pick it up if we have available capacity.
    if sum(inventory) < backpack_capacity:
        pickup_entry = None
        pickup_idx = None
        for entry in remaining_res:
            x,y,r = entry #Extract the coordinates and resource type from the entry from the remaining resources element from the current state.
            if (x,y) == successor: #We found a resource at the successor tile.
                i = resource_id[r] #Get the index of the resource type.
                #We check if we still need this resource to meet the goal.
                if inventory[i] + returned_base_res[i] < goal[resource_order[i]]:
                    pickup_entry = entry
                    pickup_idx = i
                    break
        #If the pickup_entry is not None, we pick up the resource because it was valid:
        if pickup_entry is not None:
            inventory[pickup_idx] += 1
            remaining_res.remove(pickup_entry)


    #Update the datastructure
    new_state = State(
        loc_agent = successor,
        contents=tuple(inventory),
        returned_base=tuple(returned_base_res),
        remaining=frozenset(remaining_res) )
        
    return new_state, move_cost


#4) 
def manhattan_distance(current: State):

    if current.loc_agent == (0,0) and goal_met(current.returned_base):
        return 0
    
    #Calculate the total number of resources still needed to meet the goal. 
    need = []
    for i, r in enumerate(resource_order):
        need.append(max(0, goal[r] - (current.returned_base[i] + current.contents[i])))
    total_need = sum(need)

    #Current distance to base
    x_cur, y_cur = current.loc_agent
    base_distance = abs(x_cur - 0) + abs(y_cur - 0)

    #Case 1) If we don't need to collect more resources, but we are still carrying some, we need to return to base.
    #So the heuristic returns the Manhattan distance from the current position to the base
    if total_need == 0 and sum(current.contents) > 0:
        return base_distance
        #You must at least walk back to base to finish, so distance-to-base is a hard lower bound on the remaining cost.
    
    #Case 2) We need to collect more resources.
    if total_need > 0:
        
        #2.1) If the backpack is full, we need to return to base to deposit resources before collecting more.
        if sum(current.contents) >= backpack_capacity:
            return base_distance
        
        #2.2) If the backpack is not full, we can try to collect more resources.
        best = float('inf')
        for (x,y,r) in current.remaining:
            i = resource_id[r]
            if need[i] > 0: #We only consider resources that we still need to collect.
                distance = abs(current.loc_agent[0]-x) + abs(current.loc_agent[1]-y)
                if distance < best:
                    best = distance
        
        if best < float('inf'): #If we found a resource that we need to collect.
            return best 
        
        # if we didn't find a needed tile (shouldn’t happen if total_need>0), fallback to base
        return base_distance

    #Otherwise, we don't need any additional resource and we are not carrying any, so we go back to base.
    return base_distance

#5) Zero heuristic for comparison.
def heuristic_zero(current: State):
    z = 0
    return z  # Dijkstra baseline

#6)
def reconstruct_path(parents, goal_state):
    #Follow parent pointers back to start; return the path as a list of States.
    path = [goal_state] #Start from the goal state (found solution)
    cur = goal_state
    while cur in parents:
        cur = parents[cur]
        path.append(cur)
    path.reverse()
    return path



#--------------------------------------------------------------------------------------------------------
#A* Algorithm Implementation
#--------------------------------------------------------------------------------------------------------

def A_Star_Algorithm(heuristic_fn = None):

    start = make_start_state() #Initial state of the agent.

    #Initialization of the A* algorithm variables.
    Frontier = { start: heuristic_fn(start) } #Frontier is a dictionary with the state as key and f_n as value.
    g_score = {start: 0.0}
    origin_node_info = {}
    Explored = set()

    expansions = 0 #Counter for the number of expansions.
    t_start = time.time() #Start time for measuring execution time.

    while Frontier: #Run until there are no more nodes to explore and the goal is not reached.
        
        current_node = min(Frontier, key=Frontier.get)  #Node with the lowest h_n + g_n.
        Frontier.pop(current_node) #Remove the current node from the frontier.
        Explored.add(current_node) #Add the current node to the explored set.
        expansions += 1 # count a node expansion when we pop and expand it

        if current_node.loc_agent == (0,0) and goal_met(current_node.returned_base): #Goal test: we are at the base and have met the goal.
            t_end = time.time() #End time for measuring execution time.
            path_states = reconstruct_path(origin_node_info, current_node)
            path_length = max(0, len(path_states)-1) #Number of steps taken (edges in the path).
            total_cost = g_score[current_node] #Total cost to reach the goal.
            runtime_ms = int((t_end - t_start) * 1000) #Runtime in milliseconds.

            print("Goal reached! All of the resources were collected and returned to the starting point.")
            return {
                "found_solution": True,
                "path_length": path_length,
                "expanded": expansions,
                "runtime_ms": runtime_ms,
                "cost": total_cost,
                "goal_state": current_node,
                "path_states": path_states,
                "path_positions": [s.loc_agent for s in path_states],
            }

        parent_g = g_score[current_node]
    
        neighbors = list(get_successors(current_node.loc_agent))
        for s in neighbors:
            new_state, move_cost = step(current_node, s)
            if new_state in Explored:
                continue
        
            #Calculate g and f scores for the successor.
            tentative_g = parent_g + move_cost
            
            #We only keep the best path:
            if tentative_g < g_score.get(new_state, float('inf')):
                g_score[new_state] = tentative_g # If the cost to reach new_state is lower, we update g_score because we found a better path.
                f_new = tentative_g + heuristic_fn(new_state) #Gives us f_n = g_n + h_n. Current best estimate of total cost through new_state + heuristic.
                Frontier[new_state] = f_new 
                origin_node_info[new_state] = current_node # We update the parent pointer to reconstruct the path later on success.

    # failure
    t_end = time.time()
    return {
        "found_solution": False,
        "path_length": 0,
        "expanded": expansions,
        "runtime_ms": int((t_end - t_start) * 1000),
        "cost": math.inf,
        "goal_state": None,
        "path_states": [],
        "path_positions": [],
    }


#--------------------------------------------------------------------------------------------------------
#Functions to visualize the results
#--------------------------------------------------------------------------------------------------------

def compute_visit_times(path_positions):
#Build a dict: (x,y) -> [t0, t1, ...] where we store the times when we ENTER that cell. Start position gets time 0.
    times = {} #Dictionary to store the visit times for each cell.
    if not path_positions: #If the path is empty, return empty times (model failed to find a solution).
        return times
    # time at start
    t = 0.0
    x0, y0 = path_positions[0] #Start position
    times.setdefault((x0,y0), []).append(round(t, 2))
    # accumulate step per move
    for a, b in zip(path_positions[:-1], path_positions[1:]): #For each pair of consecutive positions in the path.
        t +=  1 #next step
        xb, yb = b #We record the time when we enter cell b.
        times.setdefault((xb, yb), []).append(t) 
    return times


def plot_board_with_times(terrain_grid, times_by_cell, title="Board with visit times",resource_names=('stones','irons','crystals')):
    """
Draws the 5×5 game board and colors each cell based on its terrain type (grassland, hills, swamp, mountain). 
On top of that, it writes the times we visited each cell (like t=0, t=3, ...), so you can see when and how often we passed through. 
It also lets you overlay resource markers—just pass which ones you want in resource_names.
Basically, this gives a quick visual of the terrain plus our path timing and where the loot is.
    """
    terrain_to_code = {'grassland':0, 'hills':1, 'swamp':2, 'mountain':3} # Mapping terrain types to color codes.
    H = len(terrain_grid) #Height of the grid.
    W = len(terrain_grid[0]) #Width of the grid.
    grid_codes = [[terrain_to_code[terrain_grid[i][j]] for j in range(W)] for i in range(H)] # We convert the terrain grid to a grid of color codes.

    cmap = ListedColormap([
        "#a6d96a",  # grassland green
        "#fdae61",  # hills orange
        "#74add1",  # swamp blue
        "#bdbdbd",  # mountain gray
    ])

    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(grid_codes, cmap=cmap, origin='upper') # Display the terrain grid as an image (top-left is (0,0))

    # grid lines + labels
    ax.set_xticks(range(W)); ax.set_yticks(range(H)) # put tick marks at every column (x) and row (y) index so the grid is labeled 0..W-1 and 0..H-1
    ax.set_xticks([x-0.5 for x in range(1,W)], minor=True) # add minor x-ticks halfway between cells (at .5) so we can draw the vertical grid lines
    ax.set_yticks([y-0.5 for y in range(1,H)], minor=True) # add minor y-ticks halfway between cells (at .5) so we can draw the horizontal grid lines
    ax.grid(which='minor', color='black', linewidth=1) # turn on the minor-grid (the cell boundaries) in black with a 1px line
    ax.set_xticklabels(range(W)); ax.set_yticklabels(range(H)) # label the major ticks with their index numbers so it’s easy to read coordinates
    ax.set_title(title) # set the title at the top of the plot to whatever we pass in


    # annotate visit times
    for i in range(H):
        for j in range(W):
            tlist = times_by_cell.get((i,j), []) # grab the list of times we entered cell (i,j); default to empty if never visited
            if tlist:  #only draw text if we actually visited this cell
                txt = ", ".join(str(t) for t in tlist) # turn the list of times into a comma-separated string like "0, 3, 7"
                ax.text(j, i, txt, va='center', ha='center', fontsize=6) # place the text in the middle of the cell (x=j, y=i)

    # base marker at (0,0)
    ax.scatter(0, 0 + 0.25, s=180, marker='*', color='black', label='base')


    # overlay resources 
    style = {    # define how each resource type should look on the plot
        'stones':   dict(marker='o', edgecolor='red',    facecolor='none', label='stones'),     # stones = red circles 
        'irons':    dict(marker='s', edgecolor='blue',   facecolor='none', label='irons'),      # irons  = blue squares 
        'crystals': dict(marker='^', edgecolor='pink', facecolor='none', label='crystals'),   # crystals = pink triangles
    }

    handles = [                                                         # legend entries for the terrain colors + base icon
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor="#a6d96a", markersize=10, label='grassland'),  # green square for grassland
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor="#fdae61", markersize=10, label='hills'),      # orange square for hills
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor="#74add1", markersize=10, label='swamp'),      # blue square for swamp
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor="#bdbdbd", markersize=10, label='mountain'),   # gray square for mountain
        plt.Line2D([0],[0], marker='*', color='black', markersize=10, label='base'),                               # black star for the base
    ]

    for rname in resource_names: 
        if rname not in resources: # if they passed a name we don't have, just skip it
            continue
        st = style[rname]  # grab the plotting style for this resource type

        # draw each resource loc_agent; note (row, col) => scatter(x=col, y=row)
        for (rx, ry) in resources[rname]:                          
            ax.scatter(ry, rx + 0.25,s=180, marker=st['marker'],facecolors=st['facecolor'], edgecolors=st['edgecolor'], linewidths=2)  # place a marker at that cell (x=j=col, y=i=row), nudged up a bit
            ax.text(ry, rx + 0.25, rname[0].upper(), color=st['edgecolor'], ha='center', va='center',fontsize=10, fontweight='bold') # draw a single letter (S/I/C) on top of the marker
        handles.append(plt.Line2D([0],[0], marker=st['marker'], color='w', markerfacecolor='none',markeredgecolor=st['edgecolor'],markersize=8, label=rname))# add this resource’s marker to the legend as well
                                
                                
    ax.legend(handles=handles, loc='upper right', fontsize=8, frameon=True) 
    plt.tight_layout()                                                        
    plt.show()      #show the figure                                                        




def plot_all_maps_with_paths():
    global terrain_grid
    for grid_name, grid_def in TEST_MAPS.items():
        terrain_grid = grid_def
        res = A_Star_Algorithm(heuristic_fn=manhattan_distance) # Run A* with the Manhattan heuristic.
        path_positions = res["path_positions"]
        times_by_cell = compute_visit_times(path_positions)
        title = f"{grid_name} | cost={res['cost']} | steps={res['path_length']}"
        # show ALL resources:
        plot_board_with_times(terrain_grid, times_by_cell, title, resource_names=('stones','irons','crystals'))



#----------------------------------------------------------------------------------------------------------------------------------
#Run
#----------------------------------------------------------------------------------------------------------------------------------

def run_all_tests():
    global terrain_grid

    heuristics = [
        ("Heuristic Zero", heuristic_zero),
        ("Manhattan distance heuristic", manhattan_distance), 
    ]

    print("\n=== Results Table ===")
    print("Grid, Heuristic, Found, PathLen, Expanded, Runtime(ms), Cost")

    for grid_name, grid_def in TEST_MAPS.items():
        terrain_grid = grid_def 
        for hname, heuristic in heuristics:
            res = A_Star_Algorithm(heuristic_fn=heuristic)
            print(f"{grid_name}, {hname}, {res['found_solution']}, {res['path_length']}, {res['expanded']}, {res['runtime_ms']}, {res['cost']}")

if __name__ == "__main__":
    run_all_tests()
    plot_all_maps_with_paths()






