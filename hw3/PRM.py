# assume all the polygons on the same size
from dis import dis
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import shapely
from shapely.geometry import LineString, Point
import GraphSearch
from math import dist


random.seed(20)
X_LIMIT_LEFT = 0
X_LIMIT_RIGHT = 300

Y_LIMIT_DOWN = 0
Y_LIMIT_UP = 300
N = 10
X_Obs = 15
Y_Obs = 10


def is_legal(x,y) -> bool:
    if x < X_LIMIT_LEFT:
        return False
    if x + X_Obs > X_LIMIT_RIGHT:
        return False
    if y < Y_LIMIT_DOWN:
        return False
    if y + Y_Obs > Y_LIMIT_UP:
        return False
    return True

class Obstacle(object):
    def __init__(self,x_left: float, y_left: float):
        # check the border
        if is_legal(x_left, y_left):
            self.x_left = x_left
            self.y_left = y_left
            self.x_right = x_left + X_Obs
            self.y_right = y_left + Y_Obs

    def _get_left_button(self):
        return (self.x_left, self.y_left)

    def _get_right_button(self):
        return (self.x_right, self.y_left)

    def _get_left_up(self):
        return (self.x_left, self.y_right)

    def _get_right_up(self):
        return (self.x_right, self.y_right)

    def get_lines(self) -> list:
        """
        :return: list of lines
        """
        line_1 = LineString([self._get_left_button(), self._get_right_button()])
        line_2 = LineString([self._get_right_button(), self._get_right_up()])
        line_3 = LineString([self._get_right_up(), self._get_left_up()])
        line_4 = LineString([self._get_left_up(), self._get_left_button()])
        return [line_1, line_2, line_3, line_4]



class Node(object):
    def __init__(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        # self.near_node = None

    def point(self):
        """
        :return: (x,y)
        """
        return (self.x_pos, self.y_pos)

    def __str__(self):
        return f'{self.x_pos}_{self.y_pos}'



class PRM(object):
    def __init__(self, thd, nodes_number, obstacles_list):
        self.thd = thd
        self.nodes_number = nodes_number
        self.forest = dict()

        self.obstacles_list = obstacles_list

    def get_avg_node_degree(self):
        sum_nodes = 0
        for adj_list in self.forest.values():
            sum_nodes += len(adj_list)
        number_of_nodes = len(self.forest)
        avg_node_degree = sum_nodes/number_of_nodes
        return avg_node_degree

    def plot_obstacles(self):
        fig, ax = plt.subplots()
        # add obstacles
        for obstacle in self.obstacles_list:
            ax.add_patch(Rectangle((obstacle.x_left, obstacle.y_left), X_Obs, Y_Obs,
                                   edgecolor='blue', lw=1, fill=True, facecolor='red'))

        plt.xlim([X_LIMIT_LEFT, X_LIMIT_RIGHT])
        plt.ylim([Y_LIMIT_DOWN, Y_LIMIT_UP])
        plt.show()

    def plot_nodes(self):
        fig, ax = plt.subplots()
        # add obstacles
        print(self.forest)
        #     ax.add_patch(Rectangle((obstacle.x_left, obstacle.y_left), X_Obs, Y_Obs,
        #                            edgecolor='blue', lw=1, fill=True, facecolor='red'))
        #
        # plt.xlim([X_LIMIT_LEFT, X_LIMIT_RIGHT])
        # plt.ylim([Y_LIMIT_DOWN, Y_LIMIT_UP])
        # plt.show()
        X_list = [float(key.split('_')[0]) for key in self.forest.keys()]
        Y_list = [float(key.split('_')[1]) for key in self.forest.keys()]

        plt.scatter(X_list, Y_list, marker="o")

    def plot_nodes_with_edges(self):
        plt.figure(figsize=(36, 36))
        G = nx.Graph()
        # add nodes
        [G.add_node(key, pos=(float(key.split('_')[0]), float(key.split('_')[1]))) for key in self.forest.keys()]
        # add edges
        [[G.add_edge(key, self.forest[key][i]) for i in range(0, len(self.forest[key]))]
                    for key in self.forest.keys()]


        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos)
        plt.show()

    def plot_all(self):
        f, ax = plt.subplots(1, 1, figsize=(28,28))

        G = nx.Graph()
        # add nodes
        [G.add_node(key, pos=(float(key.split('_')[0]), float(key.split('_')[1]))) for key in self.forest.keys()]

        # [G.add_node(str(index), pos=(float(key.split('_')[0]), float(key.split('_')[1])))
        # for index, (key, value) in enumerate(self.forest.items())]
        # add edges
        [[G.add_edge(key, self.forest[key][i]) for i in range(0, len(self.forest[key]))]
                     for key in self.forest.keys()]

        pos = nx.get_node_attributes(G, 'pos')
        # get pos coordinate
        pos_coordinate = [pos[key] for key in pos]

        nx.draw(G, pos, with_labels=True,font_color='r')


        # add obstacles
        for obstacle in self.obstacles_list:
            ax.add_patch(Rectangle((obstacle.x_left, obstacle.y_left), X_Obs, Y_Obs,
                                   edgecolor='blue', lw=1, fill=True, facecolor='red'))
        plt.show()

    def add_node(self,node:Node) -> bool:
        # check if the node not in "bad" area
        for obstacle in self.obstacles_list:
            if (obstacle.x_left <= node.x_pos <= obstacle.x_right and
                    obstacle.y_left <= node.y_pos <= obstacle.y_right):
                return False

        # crate node list if is not exsist
        if not str(node) in self.forest.keys():
            self.forest[str(node)] = []

        nearest_neighbor_list = self._nearest_neighbors(node)

        # nearest not dont exist
        if len(nearest_neighbor_list) == 0:
            return True

        # add node to the forest
        for neighbor in nearest_neighbor_list:
            # add neighbor to node list
            if str(neighbor) not in self.forest[str(node)]:
                self.forest[str(node)].append(str(neighbor))

            # crate neighbor list if is not exsist
            if not str(neighbor) in self.forest.keys():
                self.forest[str(neighbor)] = []

            # add node to neighbor list
            if node not in self.forest[str(neighbor)]:
                self.forest[str(neighbor)].append(str(node))

        # if not str(nearest_neighbor) in self.forest[str(nearest_neighbor)]:
        #    self.forest[str(nearest_neighbor)] = []
        # self.forest[str(nearest_neighbor)].append(str(node))

        return True

    def _check_reachable(self,node, neighbor):
        # create line from node to neighbor
        #line = LineString([(0, 0), (1, 1)])
        #other = LineString([(0, 1), (1, 0)])
        # print(line.intersects(other))
        line_node_neighbor = LineString([node.point(), neighbor.point()])
        for obstacle in self.obstacles_list:
            for line in obstacle.get_lines():
                if line_node_neighbor.intersects(line):
                    # lines are intersects somewhere
                    print('lines are interpolate')
                    return False

        return True

    def _nearest_neighbors(self, node:Node) -> list:

        # TODO How do I prevent the passage of an
        #  edge within an area that must not be crossed?
        min_distance = np.Inf
        nearest_neighbor = []
        for neighbor in self.forest.keys():
            neighbor_x_pos = neighbor.split('_')[0]
            neighbor_y_pos = neighbor.split('_')[1]
            distance_x = (node.x_pos - float(neighbor_x_pos))**2
            distance_y = (node.y_pos - float(neighbor_y_pos))**2
            distance = (distance_x + distance_y)**0.5
            if  distance!=0:
            # check reaching
                reachable = self._check_reachable(node, Node(float(neighbor_x_pos), float(neighbor_y_pos)))
                if not reachable:
                    # continue to the next neighbor
                    continue
            # check distance
                if distance < self.thd:
                    nearest_neighbor.append(neighbor)
                # min_distance = distance
                # nearest_neighbor = neighbor

        return nearest_neighbor

def GeneratePRM(thd:float,nodes:int,obstacles_list:list):

    # create nodes
    prm_model = PRM(thd=thd, nodes_number=nodes, obstacles_list=obstacles_list)
    num_nodes_added = 0  
    while True:
        x_pos = random.randint(X_LIMIT_LEFT, X_LIMIT_RIGHT)
        y_pos = random.randint(Y_LIMIT_DOWN, Y_LIMIT_UP)
        node = Node(x_pos=x_pos, y_pos=y_pos)
        # add the node
        is_node_added = prm_model.add_node(node)
        print(f'the node is {is_node_added}')
        if is_node_added:
            num_nodes_added += 1
        if num_nodes_added >= prm_model.nodes_number:
            break

    # prm_model.plot_obstacles()
    prm_model.plot_all()

    # return prm
    return prm_model

def draw_configurations():
    obstacles_list = list()
    # crate obstacles Rectangle
    for i in range(0, N):
        x_left_right_pose = random.uniform(X_LIMIT_LEFT, X_LIMIT_RIGHT-X_Obs)
        y_left_right_pose = random.uniform(Y_LIMIT_DOWN, Y_LIMIT_UP-Y_Obs)
        obstacle = Obstacle(x_left_right_pose, y_left_right_pose)
        obstacles_list.append(obstacle)

    GeneratePRM(thd=50, nodes=100, obstacles_list=obstacles_list)

def nearest_neighbor(pos, prm_model):
    min_distance = float("inf")
    nearest_neighbor = None
    for node in prm_model.forest:
        curr_dist = dist(pos, GraphSearch.str_node_to_float_node(node))
        if curr_dist < min_distance:
            min_distance = curr_dist
            nearest_neighbor = node

    return nearest_neighbor

def plot_shortest_path(shortest_path:list,forest:dict,obstacles_list:list):
    f, ax = plt.subplots(1, 1, figsize=(28, 28))

    G = nx.Graph()
    # add nodes
    [G.add_node(key, pos=(float(key.split('_')[0]), float(key.split('_')[1]))) for key in forest.keys()]

    # add edges
    [[G.add_edge(key, forest[key][i],color='black',weight=1) for i in range(0, len(forest[key]))]
    for key in forest.keys()]

    # add shortest path edges (with color)
    [G.add_edge(shortest_path[index], shortest_path[index+1],color='g',weight=6) for index in
      range(0, len(shortest_path)-1)]

    # get params to plot
    weights = nx.get_edge_attributes(G, 'weight').values()
    colors = nx.get_edge_attributes(G, 'color').values()
    pos = nx.get_node_attributes(G, 'pos')
    # get pos coordinate
    pos_coordinate = [pos[key] for key in pos]

    for obstacle in obstacles_list:
        ax.add_patch(Rectangle((obstacle.x_left, obstacle.y_left), X_Obs, Y_Obs,
                               edgecolor='blue', lw=1, fill=True, facecolor='red'))

    nx.draw(G, pos, with_labels=True, font_color='r',edge_color=colors, width=list(weights))
    plt.show()

    pass

if __name__ == '__main__':
    # define Matplotlib figure and axis
    # fig, ax = plt.subplots()
    # ax.plot([0, 10], [0, 10])
    #
    # ax.add_patch(Rectangle((1, 1), 2, 6,color='red'))
    # plt.xlim([X_LIMIT_LEFT, X_LIMIT_RIGHT])
    # plt.ylim([Y_LIMIT_DOWN, Y_LIMIT_UP])
    # plt.show()
    # prm = draw_configurations()
    # ob = Obstacle(50, 50)

    ### For Ron
    print('ron')
    obstacles_list = list()
    # crate obstacles Rectangle
    for i in range(0, N):
        x_left_right_pose = random.uniform(X_LIMIT_LEFT, X_LIMIT_RIGHT-X_Obs)
        y_left_right_pose = random.uniform(Y_LIMIT_DOWN, Y_LIMIT_UP-Y_Obs)
        obstacle = Obstacle(x_left_right_pose, y_left_right_pose)
        obstacles_list.append(obstacle)

    prm_model = GeneratePRM(thd=50, nodes=100, obstacles_list=obstacles_list)

    # part b
    start_pos = nearest_neighbor((0, 0), prm_model)
    goal_pos = nearest_neighbor((X_LIMIT_RIGHT, Y_LIMIT_UP), prm_model)

    # solve graph search problem
    dijksta_solver = GraphSearch.Dijkstra(prm_model.forest)
    dijksta_solver.compute_costs(start_pos)
    shortest_path = dijksta_solver.find_path(goal_pos) # path is reversed
    print(shortest_path)
    plot_shortest_path(shortest_path=shortest_path,forest=prm_model.forest,obstacles_list=obstacles_list)