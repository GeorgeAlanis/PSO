import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

def sortByX(element, reverse=False):
    '''
        In situ sort (more or less) by X attr.
        INPUT: list or np.array of Point2D class.
        RETURN: void.
    '''
    if not isinstance(element, list) and type(element) != np.array:
        raise ValueError('Must be list or array.')
    return element.sort(key=lambda x: (x.X, x.Y), reverse=reverse)

def sortByY(element, reverse=False):
    '''
        In situ sort (more or less) by y element.
        INPUT: list or np.array of Point2D class.
        RETURN: void.
    '''
    if not isinstance(element, list) and type(element) != np.array:
        raise ValueError('Must be list or array.')
    return element.sort(key=lambda x: (x.Y, x.X), reverse=reverse)

def turnRight(points, point2):
    '''
        INPUT:
            - point, list of two point2D, just the last two elements are used.
            - point2, Point2D instance, the last one
        RETURN: boolean, true if it represents a turn right.
    '''
    return ((points[-1].Y-points[-2].Y)*(point2.X-points[-2].X)) > ((point2.Y-points[-2].Y)*(points[-1].X-points[-2].X))

def turnLeft(points, point2):
    '''
        INPUT:
            - point, list of two point2D, just the last two elements are used.
            - point2, Point2D instance, the last one
        RETURN: boolean, true if it represents a turn right.
    '''
    return ((points[-1].Y-points[-2].Y)*(point2.X-points[-2].X)) < ((point2.Y-points[-2].Y)*(points[-1].X-points[-2].X))


def getConvexHull(elements):
    # '''
    #     INPUT: list of Point2D instances
    #     RETURN: list whit the points in the ConvexHUll of the set
    # '''
    # sortByX(elements)
    # upper = elements[:2]
    # lower = elements[:2]
    # for i in xrange(2, len(elements)):
    #     while len(upper) > 1 and not turnRight(upper[-2:], elements[i]):
    #         upper.pop()
    #     while len(lower) > 1 and not turnLeft(lower[-2:], elements[i]):
    #         lower.pop()
    #     upper.append(elements[i])
    #     lower.append(elements[i])
    # sortByX(lower, reverse=True)
    # return upper + lower[1:-1]
    return ConvexHull(np.array([[x.X,x.Y] for x in elements])).vertices

def getConvexPolygonArea(elements):
    # '''
    #     INPUT: list of Point2D instances in clockwise order.
    #     RETURN: Area of the polygon represented by those points.
    # '''
    # area = elements[-1].Y * elements[0].X - elements[-1].X * elements[0].Y
    # for i in xrange(len(elements)-1):
    #     area -= elements[i].X * elements[i+1].Y
    #     area += elements[i+1].X * elements[i].Y
    # return area/2.
    return ConvexHull(np.array([[x.X,x.Y] for x in elements])).area

def getData(file_path='../assets/Sprint7ToroideMixto.csv'):
    data = pd.read_csv(file_path)
    data['tiempo_en_tienda'] = data['demanda'] * data['frecuencia']
    stores = np.array(list(zip(data.lat.values, data.lon.values, data.tiempo_en_tienda.values)))
    return stores

def euclidean(p1, p2):
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])
