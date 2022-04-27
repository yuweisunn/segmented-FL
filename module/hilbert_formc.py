# -*- coding: utf-8 -*-

from module.hilbert import *

def hilbert_formc(size):
    
    pmax = size
    side = 2**pmax
    min_coord = 0
    max_coord = side - 1
    cmin = min_coord - 0.5
    cmax = max_coord + 0.5
    
    offset = 0
    dx = 0.5
    
    point_list = []
    
    for p in range(pmax, 0, -1):
        hc = HilbertCurve(p, 2)
        sidep = 2**p
    
        npts = 2**(2*p)
        pts = []
        for i in range(npts):
            pts.append(hc.coordinates_from_distance(i))
        pts = [
            [(pt[0]*side/sidep) + offset,
             (pt[1]*side/sidep) + offset]
            for pt in pts]
        for i in range(npts):
            point_list.append([int(pts[i][0]),int(pts[i][1])])
            if i == 4**pmax-1:
               return point_list
    
