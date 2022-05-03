from math import dist
from queue import PriorityQueue


#Implement Node class and maybe graph class
class Dijkstra:
    def __init__(self) -> None:
        self.open = None
        self.close = None
        self.G = []
    
    def solve(self, start_node):
        self.open = PriorityQueue()
        self.open.put((0, start_node))
        self.close = []

        while not self.open.empty():
            curr_node_cost, curr_node = self.open.get()
            self.close.append(curr_node)

            for next_node in expand_node(curr_node):
                if next_node not in self.close:
                    next_node_old_cost = G[next_node]
                    next_node_new_cost = curr_node_cost + dist(curr_node, next_node)
                    if next_node_new_cost < next_node_old_cost:
                        self.open.put((next_node_new_cost, next_node))
                        G[next_node] = next_node_new_cost