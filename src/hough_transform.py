import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import itertools
import math

# step 1. store pixels in 1D array
#z_points = Alt
#x_points = u.magnitude
#y_points = v.magnitude
z_values = np.linspace(0,6.28,20)
x_values = 3 * np.cos(z_values) +5
y_values = 5 * np.sin(z_values) +5
data = np.column_stack((x_values,y_values))

plt.scatter(x_values, y_values,z_values)
plt.show()

#use on pairs found with detectEllipse
def calc_ellipse(x, y, x_centre, y_centre, rotation, semi_major, semi_minor):
    ...
    term1 = (((x - x_centre) * np.cos(rotation) + 
        (y - y_centre) * np.sin(rotation)))**2
    term2 = (((x - x_centre) * np.sin(rotation) -
        (y - y_centre) * np.cos(rotation)))**2
    ellipse = ((term1 / semi_major**2) + (term2 / semi_minor**2)) <= 1
    return ellipse

#ellipse = calc_ellipse(x, y, x0, y0, 0, a, b)
#plt.imshow(ellipse, origin="lower")  # Plot
def checkPairs(x_values, y_values):
    data = list(zip(x_values,y_values))
    pairs = [] #valid pairs
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if checkMinDist(data[i],data[j]):
                pairs.append([data[i],data[j]])
    return pairs
        
    

#step 4. check for valid pairs
def checkMinDist(x,y):
    valid = False
    dist = math.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
    if dist > 4:
        valid = True
    return valid

#valid pairs in form pairs[][][]
#first refers to the pair(0-len(pairs))
#second refers to x1y1 or x2y2 (0 or 1)
#third refers to x or y (0 or 1)
pairs = checkPairs(x_values,y_values)
print('done')
print(pairs)
minMinorEllipse = 1



def detectEllipse(pairs, data):
     minThreshold = 0
     highestVotes = 0
     global b, gxy3
     
    
     #step 2. clear accumulator array
     accumulator = np.zeros(len(pairs))
     ellipses = []
     # step 5. use equations 1-4
     for p in pairs:
         x0 = (p[0][0] + p[1][0])/2 # (1) x of center 
         y0 = (p[0][1] + p[1][1])/2 # (2) y of center
         xy0 = (x0,y0)
         xy1 = (p[0][0], p[0][1])
         xy2 = (p[1][0], p[1][1])
         a = math.sqrt((p[1][0]-p[0][0])**2 + (p[1][1]-p[0][1])**2)/2 # (3) half-length of major axis
         alpha = math.atan((p[1][1]-p[0][1])/(p[1][0]-p[0][0])) # (4) alpha/ orientation off the ellipse

         
         # step 6. Find 3rd point
         for count, xy3 in enumerate(data):
             gxy3 = xy3
             d = math.dist(xy0, xy3)
             dd = d**2
             if d >= a or d < minMinorEllipse:
                 continue
             
             # calculate Minor Axis
             aa = a**2
             f = math.dist(xy2, xy3)
             ff = f**2
             
             # step 7. Calculate length of minor axis
             cosT = (aa + dd - ff)/(2*a*d) # (6)
             coscosT = cosT**2
             
             bb = aa*dd*(1-coscosT)/(aa-dd*coscosT+0.00001) # (5)
             b = round(math.sqrt(abs(bb)))
             
             # step 8. increment the accumulator
             if b > 1:
                 accumulator[b] += 1
                
             # step 12. remove point from data
             np.delete(data,count)   
     
         # step 9. loop through all xy3
        
         # step 10. find max in accumulator array
         idx = max(accumulator)
    
         # adjust minimum threshold
         if idx > highestVotes:
             highestVotes = idx
             minThreshold = 0.99*highestVotes
  
         # step 11. output ellipse parameters
         if idx > minThreshold:
             ellipses.append([x0,y0,a,alpha,b])
            
         # step 13. clear accumulator array
         accumulator = np.zeros(len(pairs))
         
    # step 14. loop for all pixel pairs     
     
     return ellipses   
             
             
         
         
             
                
ellipses = detectEllipse(pairs, data)
for i in ellipses:
    print('possible ellipse: ', i)