# assume all the polygons on the same size
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
random.seed(20)
X_LIMIT_LEFT = 0
X_LIMIT_RIGHT = 300

Y_LIMIT_DOWN = 0
Y_LIMIT_UP = 300
N = 100
X_Obs = 15
Y_Obs =  10
def is_legal(x,y) ->bool:
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

class Node(object):
    def __init__(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.near_node = None


    def __eq__(self,other):
        if self.x_pos == other.x_pos and self.y_pos == other.y_pos:
            return True
        return False

    def __str__(self):
        return f'{self.x_pos}_{self.y_pos}'



class PRM(object):
    def __init__(self, thd, nodes_number, obstacles_list):
        self.thd = thd
        self.nodes_number = nodes_number
        self.forest = dict()
        self.obstacles_list = obstacles_list

    def plot_obstacles(self):
        fig, ax = plt.subplots()
        # add obstacles
        for obstacle in self.obstacles_list:
            ax.add_patch(Rectangle((obstacle.x_left, obstacle.y_left), X_Obs, Y_Obs,
                                   edgecolor='blue', lw=1, fill=True, facecolor='red'))

        plt.xlim([X_LIMIT_LEFT, X_LIMIT_RIGHT])
        plt.ylim([Y_LIMIT_DOWN, Y_LIMIT_UP])
        plt.show()

    def add_node(self,node:Node)->bool:
        # check if the node not in "bad" area
        for obstacle in self.obstacles_list:
            if (obstacle.x_left<=node.x_pos<=obstacle.x_right and
                    obstacle.y_left <= node.y_pos <= obstacle.y_right):
                return False

        # add node to the node-list
        nearest_neighbor = self._nearest_neighbors(node)

        return True


    def _nearest_neighbors(self, node:Node) -> Node:

        # TODO How do I prevent the passage of an
        #  edge within an area that must not be crossed?
        min_distance = np.Inf
        nearest_neighbor = None
        for neighbor in self.forest.keys():
            distance_x = (node.x_pos - neighbor.x_pos)**2
            distance_y = (node.y_pos - neighbor.y_pos)**2
            distance = (distance_x + distance_y)**0.5
            if distance < min_distance:
                min_distance = distance
                nearest_neighbor = neighbor



    def plot_nodes(self):

        pass

def GeneratePRM(thd:float,nodes:int,obstacles_list:list):

    # create nodes
    prm_model = PRM(thd=20, nodes_number=100, obstacles_list=obstacles_list)
    num_nodes_added = 0
    while True:
        x_pos = random.uniform(X_LIMIT_LEFT, X_LIMIT_RIGHT)
        y_pos = random.uniform(Y_LIMIT_DOWN, Y_LIMIT_UP)
        node = Node(x_pos=x_pos, y_pos=y_pos)
        # add the node
        is_node_added = prm_model.add_node(node)
        print(f'the node is {is_node_added}')
        if is_node_added:
            num_nodes_added += 1
        if num_nodes_added >= prm_model.nodes_number:
            break

    prm_model.plot_obstacles()


def draw_configurations():
    obstacles_list = list()
    # crate obstacles Rectangle
    for i in range(0, N):
        x_left_right_pose = random.uniform(X_LIMIT_LEFT, X_LIMIT_RIGHT-X_Obs)
        y_left_right_pose = random.uniform(Y_LIMIT_DOWN, Y_LIMIT_UP-Y_Obs)
        obstacle = Obstacle(x_left_right_pose, y_left_right_pose)
        obstacles_list.append(obstacle)


    GeneratePRM(thd=20, nodes=100, obstacles_list=obstacles_list)


if __name__ == '__main__':
    # define Matplotlib figure and axis
    # fig, ax = plt.subplots()
    # ax.plot([0, 10], [0, 10])
    #
    # ax.add_patch(Rectangle((1, 1), 2, 6,color='red'))
    # plt.xlim([X_LIMIT_LEFT, X_LIMIT_RIGHT])
    # plt.ylim([Y_LIMIT_DOWN, Y_LIMIT_UP])
    # plt.show()
    draw_configurations()
    # ob = Obstacle(50, 50)

