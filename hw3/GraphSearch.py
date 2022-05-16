from ast import Str
from math import dist
from queue import PriorityQueue
from typing import Tuple


#Implement Node class and maybe graph class
class Dijkstra:
    def __init__(self, prm_graph) -> None:
        self.prm_graph = prm_graph
        self.open = None
        self.close = None
        self.G = {node : float("inf") for node in prm_graph} # g value of each node
        self.P = {node : None for node in prm_graph} # parent of each node
    
    def compute_costs(self, start_node):
        self.open = PriorityQueue()
        self.open.put((0, start_node))
        self.G[start_node] = 0
        self.close = []

        while not self.open.empty():
            curr_node_cost, curr_node = self.open.get()
            self.close.append(curr_node)

            for next_node in self.prm_graph[curr_node]:
                if next_node not in self.close:
                    next_node_old_cost = self.G[next_node]
                    next_node_new_cost = curr_node_cost + dist(str_node_to_float_node(curr_node), str_node_to_float_node(next_node))
                    if next_node_new_cost < next_node_old_cost:
                        self.open.put((next_node_new_cost, next_node))
                        self.G[next_node] = next_node_new_cost
                        self.P[next_node] = curr_node

    def find_path_and_cost(self, goal_node):
        reversed_path = []
        curr_node = goal_node
        cost = self.G[goal_node]
        while(curr_node):
            reversed_path.append(curr_node)
            curr_node = self.P[curr_node]

            
        return reversed_path, cost

def str_node_to_float_node(node: str) -> Tuple:
    x_pos = float(node.split('_')[0])
    y_pos = float(node.split('_')[1])
    return x_pos, y_pos