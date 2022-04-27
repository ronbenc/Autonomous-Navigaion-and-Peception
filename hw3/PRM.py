# assume all the polygons on the same size

X_LIMIT_LEFT = 0
X_LIMIT_RIGHT = 200

Y_LIMIT_LEFT = 0
Y_LIMIT_RIGHT = 200

N = 10
def is_legal(x,y) ->bool:
    if x < X_LIMIT_LEFT:
        return False
    if x + N > X_LIMIT_RIGHT:
        return False
    if y < Y_LIMIT_LEFT:
        return False
    if y + N >Y_LIMIT_RIGHT:
        return False
    return True

class Obstacle(object):
    def __init__(self,x_left: float, y_left: float):
        # check the border
        if is_legal(x_left,y_left):
            self.x_left = x_left
            self.y_left = y_left



def GeneratePRM(thd:float,nodes:int,obstacles_list:list):
    pass



if __name__ == '__main__':
    ob = Obstacle(-50,-50)

