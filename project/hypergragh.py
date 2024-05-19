import numpy as np
import cv2

def distance(pixel: list ): #return the distanse to the origin
    return np.sqrt( (pixel[0])**2 + (pixel[1])**2 + (pixel[2]**2) )

def neighbor_vertex(shape: list, location: tuple, beta: int) : #give the range for neighbor_vertex
    height, width, _ = shape
    x , y = location[0] , location[1]
    return max(0,x-beta) , min(height-1, x+beta) , max(0,y-beta) , min(width-1, y+beta)

def edge(img: np.ndarray, location_x0: tuple, beta: int, mu: int): #output the hyperedge set

    x_start, x_end, y_start, y_end = neighbor_vertex(img.shape, location_x0, beta)

    e = set()
    for x in range(x_start, x_end+1):
        for y in range(y_start, y_end+1):
            location_xi = (x,y)
            if any( [ abs(img[location_x0][channel] - img[location_xi][channel]) < (2/3)*mu for channel in range(3)] )\
                and abs(distance(img[location_x0]) - distance(img[location_xi]) < mu):
                e.add(location_xi)
    return e

def HG_dilation(img ,beta=1, mu=250):

    height, width, _ = img.shape
    output = np.zeros(img.shape)

    for i in range(height):
        for j in range(width):
            location_x0 = (i,j)
            hyperedge = edge(img, location_x0, beta, mu)
            
            max_dist=0
            candi = (0,0)
            for v in hyperedge:
                if distance(img[v]) > max_dist:
                    max_dist = distance(img[v])
                    candi = v
            output[i,j] = img[candi]
    
    return output

def HG_erosion(img ,beta=1, mu=250):

    height, width, _ = img.shape
    output = np.zeros(img.shape)

    for i in range(height):
        for j in range(width):
            location_x0 = (i,j)
            hyperedge = edge(img, location_x0, beta, mu)
            
            min_dist=10000
            candi = (0,0)
            for v in hyperedge:
                if distance(img[v]) < min_dist:
                    min_dist = distance(img[v])
                    candi = v
            output[i,j] = img[candi]
            #print(min_dist)
    
    return output

def HG_opening(img: np.ndarray) -> np.ndarray:
    return HG_dilation(HG_erosion(img))

def HG_closing(img: np.ndarray) -> np.ndarray:
    return HG_erosion(HG_dilation(img))

def test():

    lena = cv2.imread('./images/samples/lena.png')

    def uniform_pepper(
        shape,
        p=0.01,
    ) -> np.ndarray:
        return np.where(np.random.sample(shape) <= p, -255, 0)

    def uniform_salt(
        shape,
        p=0.01,
    ) -> np.ndarray:
        return np.where(np.random.sample(shape) <= p, 255, 0)

    lena_salt = uniform_salt(lena.shape) + lena
    lena_pepper = uniform_pepper(lena.shape) + lena  

    cv2.imwrite('./images/outputs/hypergraph_exp/pepper.png', lena_pepper)
    cv2.imwrite('./images/outputs/hypergraph_exp/pepper_closing.png', HG_closing(lena_pepper))
    cv2.imwrite('./images/outputs/hypergraph_exp/pepper_opening.png', HG_opening(lena_pepper))
    cv2.imwrite('./images/outputs/hypergraph_exp/salt.png', lena_salt)
    cv2.imwrite('./images/outputs/hypergraph_exp/salt_closing.png', HG_closing(lena_salt))
    cv2.imwrite('./images/outputs/hypergraph_exp/salt_opening.png', HG_opening(lena_salt))

#test()

