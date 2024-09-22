import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage

class CubeTower:
    def __init__(self, configuration, parent=None):
        """
        Initializes the cube tower with a given configuration.
        :param configuration: A list of the front-facing colors of the cubes in the tower, starting from the bottom.
        :param parent: The parent node of the current node. (can be used for tracing back the path)
        """
        self.order = ['red', 'blue', 'green','yellow']
        self.configuration = configuration
        self.height = len(configuration)
        self.parent = parent

    def visualize(self):
        """
        Visualizes the current state of the cube tower showing only the front-facing side.
        """
        fig, ax = plt.subplots()
        cube_size = 1  # Size of the cube

        for i, cube in enumerate(self.configuration):
            # Draw only the front-facing side of the cube
            color = cube
            rect = plt.Rectangle((0.5 - cube_size / 2, i), cube_size, cube_size, color=color)
            ax.add_patch(rect)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.show()

    def visualize_path(self):
        """
        Visualizes the path taken to reach this state from the initial state.
        """
        path = self.get_path()
        fig, ax = plt.subplots(figsize=(len(path) * 2, self.height))
        cube_size = 1

        for i, configuration in enumerate(path):
            for j, cube in enumerate(configuration):
                color = cube
                rect = plt.Rectangle((i * (cube_size + 0.1), j), cube_size, cube_size, color=color)
                ax.add_patch(rect)

        ax.set_xlim(0, len(path) * (cube_size + 0.1))
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.show()
    
    def visualize_bidirectional_path(self, path_forward, path_backward):
        path_backward.reverse()
        full_path = path_forward[:-1] + path_backward 
        fig, ax = plt.subplots(figsize=(len(full_path) * 2, self.height))
        cube_size = 1
        for i, tower in enumerate(full_path):
            for j, cube_color in enumerate(tower.configuration):
                rect = plt.Rectangle((i * (cube_size + 0.1), j), cube_size, cube_size, color=cube_color)
                ax.add_patch(rect)
        ax.set_xlim(0, len(full_path) * (cube_size + 0.1))
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.show()

    def get_path(self):
        """
        Retrieves the path taken to reach this state from the initial state.
        """
        path = [self.configuration]
        current = self
        while current.parent is not None:
            current = current.parent
            path.append(current.configuration)
        path.reverse()
        return path
    
    def check_cube(self):
        """
        Check if the cube tower is solved, i.e. all cubes are of the same color.
        """
        return len(set(self.configuration)) == 1

    def rotate_cube(self, ind, hold_index=None):
        """
        Rotates a cube and all cubes above it, or up to a held cube.
        :param index: The index of the cube to rotate.
        :param hold_index: The index of the cube to hold, if any.
        """
        if hold_index is not None:
            for i in range(ind, hold_index):
                i_in_order = self.order.index(self.configuration[i])
                self.configuration[i] = (self.order[(i_in_order + 1) % len(self.order)])
                i += 1
            return self.configuration
        for i in range(ind, self.height):
            i_in_order = self.order.index(self.configuration[i])
            self.configuration[i] = (self.order[(i_in_order+1) % len(self.order)])
            i += 1
        return self.configuration


# Implement the search algorithms here
def dfs_search(tower:CubeTower, visited=None, moves=0, depth=0):
    if visited is None:         
        visited = set()
    config_tuple = tuple(tower.configuration)
    if config_tuple in visited:
        return None
    visited.add(config_tuple)
    if tower.check_cube():
        return {'solution': tower, 'moves': moves, 'depth': depth}
    for i in range(tower.height):   #i=ind
        for j in range(i + 1, tower.height + 1):  #j=hold_index
            new_tower = CubeTower(list(tower.configuration), parent=tower) 
            new_tower.rotate_cube(i, j)
            result = dfs_search(new_tower, visited, moves + 1, depth + 1)
            if result:
                return result
    return None

def bfs_search(tower: CubeTower):
    visited = set()
    queue = [(tower, 0)]
    ind_queue = 0
    while ind_queue < len(queue):
        current_tower, current_moves = queue[ind_queue]
        ind_queue += 1  # Move to the next item in the queue
        config_tuple = tuple(current_tower.configuration)
        if config_tuple not in visited:
            visited.add(config_tuple)
            if current_tower.check_cube():
                return {'solution': current_tower, 'moves': current_moves}
            for i in range(current_tower.height):
                for j in range(i + 1, current_tower.height + 1):
                    new_tower = CubeTower(list(current_tower.configuration), parent=current_tower)
                    new_tower.rotate_cube(i, j)
                    queue.append((new_tower, current_moves + 1))
    return None

def heuristic(configuration):
    """
    Heuristic: Count the number of cubes not matching the most common color.
    """
    most_common_color = max(set(configuration), key=configuration.count)
    return sum(1 for cube in configuration if cube != most_common_color)

def a_star_search(tower: CubeTower):
    visited = set()
    priority_queue = [(heuristic(tower.configuration), 0, tower)]
    ind_queue = 0
    while priority_queue:
        _, current_moves, current_tower = min(priority_queue, key=lambda x: x[0])
        priority_queue.remove((_, current_moves, current_tower))    # Remove selected element
        config_tuple = tuple(current_tower.configuration)
        if config_tuple not in visited:
            visited.add(config_tuple)
        if current_tower.check_cube():
            return {'solution': current_tower, 'moves': current_moves}
        for i in range(current_tower.height):
            for j in range(i + 1, current_tower.height + 1):
                new_configuration = list(current_tower.configuration)
                new_tower = CubeTower(new_configuration, parent=current_tower)
                new_tower.rotate_cube(i, j)
                g_score = current_moves + 1 
                h_score = heuristic(new_tower.configuration)
                f_score = g_score + h_score
                if tuple(new_tower.configuration) not in visited:
                    priority_queue.append((f_score, g_score, new_tower))
    return None

# Additional advanced search algorithm
# Iterative Deepening Depth-First Search (IDDFS)
def iddfs_search(tower: CubeTower):
    moves = 0
    def dls(current_tower: CubeTower, depth):
        if depth == 0 and current_tower.check_cube():
            return {'solution': current_tower, 'moves': moves}
        elif depth > 0:
            for i in range(current_tower.height):
                for j in range(i + 1, current_tower.height + 1):
                    new_tower = CubeTower(list(current_tower.configuration), parent=current_tower)
                    new_tower.rotate_cube(i, j)
                    found = dls(new_tower, depth -1)
                    if found:
                        return found
        return None
    for depth in range(0, 100):
        result = dls(tower, depth)
        moves += 1
        if result:
            return result
    return None

# Bidirectional Search
def bidirectional_search(tower: CubeTower):
    goal_config = [tower.configuration[0]] * tower.height
    goal_tower = CubeTower(goal_config)
    
    visited_forward = {tuple(tower.configuration)}
    visited_backward = {tuple(goal_tower.configuration)}
    
    queue_forward = [(tower, [tower])]
    queue_backward = [(goal_tower, [goal_tower])]

    while queue_forward and queue_backward:
        # Forward step
        current_forward, path_forward = queue_forward.pop(0)
        if tuple(current_forward.configuration) in visited_backward:
            # Finding matching node in backward path and concatenating paths
            matching_node = next((node for node, path in queue_backward if tuple(node.configuration) == tuple(current_forward.configuration)), None)
            if matching_node:
                path_backward = next(path for node, path in queue_backward if tuple(node.configuration) == tuple(current_forward.configuration))
                return {'solution' : (path_forward, path_backward[::-1]), 'moves' : len(path_forward + path_backward[::-1]) } # Reverse backward path for correct order
                
        # Exploring neighbors in forward direction
        for i in range(current_forward.height):
            for j in range(i + 1, current_forward.height + 1):
                new_tower = CubeTower(list(current_forward.configuration), parent=current_forward)
                new_tower.rotate_cube(i, j)
                if tuple(new_tower.configuration) not in visited_forward:
                    visited_forward.add(tuple(new_tower.configuration))
                    queue_forward.append((new_tower, path_forward + [new_tower]))

        # Backward step
        current_backward, path_backward = queue_backward.pop(0)
        if tuple(current_backward.configuration) in visited_forward:
            # Finding matching node in forward path and concatenating paths
            matching_node = next((node for node, path in queue_forward if tuple(node.configuration) == tuple(current_backward.configuration)), None)
            if matching_node:
                path_forward = next(path for node, path in queue_forward if tuple(node.configuration) == tuple(current_backward.configuration))
                return {'solution' : (path_forward, path_backward[::-1]), 'moves' : len(path_forward + path_backward[::-1]) }
                
        # Exploring neighbors in backward direction
        for i in range(current_backward.height):
            for j in range(i + 1, current_backward.height + 1):
                new_tower = CubeTower(list(current_backward.configuration), parent=current_backward)
                new_tower.rotate_cube(i, j)
                if tuple(new_tower.configuration) not in visited_backward:
                    visited_backward.add(tuple(new_tower.configuration))
                    queue_backward.append((new_tower, path_backward + [new_tower]))                    
    return None

# Testing the implementation here
    
# Example Usage
# self.order = ['red', 'blue', 'green','yellow']
initial_configurations = [
    ["red","yellow","blue"],
    ["red","yellow","blue","yellow"],
    ["blue","green","red","yellow"],
    ["blue","yellow","blue","yellow", "red"],
    ["yellow","red","blue","yellow", "green", "blue"]
]

def getPlots(results, algorithm_name):
    times = [result['time'] for result in results]
    moves = [result['moves'] for result in results]
    memory_usages = [result['memory_usage'] for result in results]

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(times, label='Time (s)')
    plt.ylabel('Time (s)')
    plt.title(f'{algorithm_name} Performance')
    plt.xticks(range(len(times)))  # Set x-axis ticks to integer indices

    plt.subplot(3, 1, 2)
    plt.plot(moves, label='Moves')
    plt.ylabel('Moves')
    plt.xticks(range(len(moves)))  # Set x-axis ticks to integer indices

    plt.subplot(3, 1, 3)
    plt.plot(memory_usages, label='Memory Usage (MiB)')
    plt.ylabel('Memory Usage (MiB)')
    plt.xticks(range(len(memory_usages)))  # Set x-axis ticks to integer indices

    plt.xlabel('Configuration Index')
    plt.tight_layout()
    plt.legend()
    plt.show()

results_dfs = []
results_bfs = []
results_a_star = []
results_iddfs = []
results_bidirectional = []

for config in initial_configurations:
    tower = CubeTower(config)
    print(f"Visualizing configuration: {config}")
    # tower.visualize()

    #~~~~~~~~DFS~~~~~~~~
    start_time = time.time()
    mem_usage_before = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)
    solution_dfs = dfs_search(tower)
    mem_usage_after = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)
    end_time = time.time()
    if solution_dfs:
        print("    -> Visualizing DFS Solution")
        solution_dfs['time'] = end_time - start_time
        solution_dfs['memory_usage'] = mem_usage_after - mem_usage_before
        results_dfs.append(solution_dfs)
        print(f"Configuration: {config}, Moves: {solution_dfs['moves']}, Time: {solution_dfs['time']:.4f}s, Memory Usage: {solution_dfs['memory_usage']:.4f} MiB")
        # solution_dfs['solution'].visualize_path()

    else:
        print("No solution of the DFS Algorithm found")

    # #~~~~~~~~BFS~~~~~~~~
    start_time = time.time()
    mem_usage_before = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)
    solution_bfs = bfs_search(tower)
    mem_usage_after = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)
    end_time = time.time()
    if solution_bfs:
        print("    -> Visualizing BFS Solution")
        solution_bfs['time'] = end_time - start_time
        solution_bfs['memory_usage'] = mem_usage_after - mem_usage_before
        results_bfs.append(solution_bfs)
        print(f"Configuration: {config}, Moves: {solution_bfs['moves']}, Time: {solution_bfs['time']:.4f}s, Memory Usage: {solution_bfs['memory_usage']:.4f} MiB")
        # solution_bfs['solution'].visualize_path()
    else:
        print("No solution of the BFS Algorithm found")

    #~~~~~~~~A*~~~~~~~~
    start_time = time.time()
    mem_usage_before = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)
    solution_a_star = a_star_search(tower)
    mem_usage_after = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)
    end_time = time.time()
    if solution_a_star:
        print("    -> Visualizing A Star Solution")
        solution_a_star['time'] = end_time - start_time
        solution_a_star['memory_usage'] = mem_usage_after - mem_usage_before
        results_a_star.append(solution_a_star)
        print(f"Configuration: {config}, Moves: {solution_a_star['moves']}, Time: {solution_a_star['time']:.4f}s, Memory Usage: {solution_a_star['memory_usage']:.4f} MiB")
        # solution_a_star['solution'].visualize_path()
    else:
        print("No solution found using A* Search")

    #~~~~~~~~IDDFS~~~~~~~~
    start_time = time.time()
    mem_usage_before = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)
    solution_iddfs = iddfs_search(tower)
    mem_usage_after = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)
    end_time = time.time()
    if solution_iddfs:
        print("    -> Visualizing IDDFS Solution")
        solution_iddfs['time'] = end_time - start_time
        solution_iddfs['memory_usage'] = mem_usage_after - mem_usage_before
        results_iddfs.append(solution_iddfs)
        print(f"Configuration: {config}, Moves: {solution_iddfs['moves']}, Time: {solution_iddfs['time']:.4f}s, Memory Usage: {solution_iddfs['memory_usage']:.4f} MiB")
        # solution_iddfs['solution'].visualize_path()
    else:
        print("No solution found using IDDFS")
        
    #~~~~~~~~Bidirectional~~~~~~~~
    start_time = time.time()
    mem_usage_before = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)
    solution_bidirectional = bidirectional_search(tower)
    mem_usage_after = memory_usage(-1, interval=0.01, timeout=1, max_usage=True)
    end_time = time.time()
    if solution_bidirectional:
        print("    -> Visualizing Bidirectional Search Solution")
        solution_bidirectional['time'] = end_time - start_time
        solution_bidirectional['memory_usage'] = mem_usage_after - mem_usage_before
        results_bidirectional.append(solution_bidirectional)
        print(f"Configuration: {config}, Moves: {solution_bidirectional['moves']}, Time: {solution_bidirectional['time']:.4f}s, Memory Usage: {solution_bidirectional['memory_usage']:.4f} MiB")
        path_forward, path_backward = solution_bidirectional['solution']
        tower.visualize_bidirectional_path(path_forward, path_backward)
    else:
        print("No solution found using Bidirectional Search")

getPlots(results_dfs, "DFS")
getPlots(results_bfs, "BFS")
getPlots(results_a_star, "A*")
getPlots(results_iddfs, "IDDFS")
getPlots(results_bidirectional, "Bidirectional")