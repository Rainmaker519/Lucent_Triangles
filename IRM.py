import torch.nn as nn
import torch
import torch.nn.functional as funky_funk
from typing import List
from Vector2 import Vector2 
import math



#Returns the point where two line segments intersect if they intersect, 
#and they aren't parallel
def get_intersection(a,b,c,d):
    ab = b - a
    cd = d - c

    ab_cross_cd = ab.cross(cd)

    #first check if ab actually intersects cd
    if Vector2.intersect(a,b,c,d):
        if ab_cross_cd == 0:
            return None
        else:
            ac = c - a
            t1 = ac.cross(cd) / ab_cross_cd
            t2 = -ab.cross(ac) / ab_cross_cd
            return a + ab.scaled(t1), t1, t2
    else:
        return None

def are_sides_equal(side1,side2):
    similar_coords = 0
    if are_points_equal(side1[0],side2[0]):
        similar_coords = similar_coords + 1
    elif are_points_equal(side1[0],side2[1]):
        similar_coords = similar_coords + 1

    if are_points_equal(side1[1],side2[0]):
        similar_coords = similar_coords + 1
    elif are_points_equal(side1[1],side2[1]):
        similar_coords = similar_coords + 1

    if similar_coords >= 2:
        print("Side 1:",side1," , Side 2:", side2)
        return True

    else:
        return False

def are_side_groups_equal(sides1,sides2):
    similar_sides = 0

    if len(sides1) == len(sides2) and (len(sides1) == 3 or len(sides1) == 4):
        for side1 in sides1:
            for side2 in sides2:

                if are_sides_equal(side1,side2):
                    similar_sides = similar_sides + 1
        
        print("len sides 1:", len(sides1), " len sides 2:", len(sides2))
        print(similar_sides)
        if similar_sides == len(sides1) and similar_sides == len(sides2):
            return True
        else:
            return False
        

    
def are_points_equal(point1:Vector2, point2:Vector2):
    if point1.x == point2.x and point1.y == point2.y:
        return True
    else:
        return False





#training model specifically to add a triangle
#   to an image to improve it's closeness score
#   to some other image
class Recreation_Model(nn.Module):
    #class 1 for 0-1 deciding rgb (.33,.33,.33)
    #class 2 for deciding opacity 0-1 being 0-100% translucent to opaque
    #class 3 for x1
    #class 5 for x2
    #class 6 for x3
    #class 7 for y1
    #class 8 for y2
    #class 9 for y3

    #only the first part so far
    #triangle selection for recreation will use a evolutionary model
    #to select from generations created by this part

    def __init__(self, num_colors = 3, num_classes = 6): #switch to 3 colors 
        super(Recreation_Model,self).__init__()

        #do 28x28 input
        self.conv1 = nn.Conv2d(in_channels = num_colors, out_channels=8*3, kernel_size=(3,3), stride=(1,1),padding=(1,1))#, groups=num_colors)

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = nn.Conv2d(in_channels = 8*3, out_channels=16*3, kernel_size=(3,3), stride=(1,1),padding=(1,1))#, groups=num_colors)

        self.fc1 = nn.Linear(16*8*8*3, num_classes)
        #will be 14x14 now
        return

    def forward(self, x):
        #NOT PART OF THE GENETIC COMPONENT
        #training to create some arbitrary number of triangles
        #which cover the darker (sum of color vals for multiple)
        #areas best
        # [how am I gonna ]
        x = funky_funk.relu(self.conv1(x))
        print(x.shape)

        x = self.pool(x)
        print(x.shape)

        x = funky_funk.relu(self.conv2(x))
        print(x.shape)

        x = self.pool(x)
        print(x.shape)

        x = x.reshape(x.shape[0], -1)
        print(x.shape)

        x = self.fc1(x)
        print(x.shape)

        return x

    # Check accuracy on training & test to see how good our model
    def check_accuracy(self, loader, device):
        num_correct = 0
        num_samples = 0
        self.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device)
                y = y.to(device=device)

                scores = self(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)


        self.train()
        return num_correct/num_samples

class Triangle():
    def __init__(self,p1,p2,p3,expanded):
        if not expanded:
            self.x1 = p1[0]
            self.y1 = p1[1]
            self.x2 = p2[0]
            self.y2 = p2[1]
            self.x3 = p3[0]
            self.y3 = p3[1]

            #all 0 at first meaning no color
            self.r: float = 0.0
            self.g: float = 0.0
            self.b: float = 0.0

            #determined after color scaling for a model run is decided
            self.greyscale: float
        else:
            self.x1 = p1[0]/32
            self.y1 = p1[1]/32
            self.x2 = p2[0]/32
            self.y2 = p2[1]/32
            self.x3 = p3[0]/32
            self.y3 = p3[1]/32

            self.r: float
            self.g: float
            self.b: float

            self.greyscale: float

        return

    def isAnyPointInsideRect(self,rect):
        for point in self.get_points_upscaled():
            if rect.isPointInside(point):
                return True
        return False

    def isPointInside(self,point):
        #takes a point and returns true if it inside the triangle being called from
        if Triangle.isInside(self.x1,self.y1,self.x2,self.y2,self.x3,self.y3,point[0],point[1]):
            return True
        else:
            return False

    
    def isInside(x1, y1, x2, y2, x3, y3, x, y):
    
       # Calculate area of triangle ABC
       A = area (x1, y1, x2, y2, x3, y3)
    
       # Calculate area of triangle PBC
       A1 = area (x, y, x2, y2, x3, y3)
        
       # Calculate area of triangle PAC
       A2 = area (x1, y1, x, y, x3, y3)
        
       # Calculate area of triangle PAB
       A3 = area (x1, y1, x2, y2, x, y)
        
       # Check if sum of A1, A2 and A3
       # is same as A
       if(A == A1 + A2 + A3):
           return True
       else:
           return False

    def get_area_overlap(self, rect):
        #figure out area of overlap of
        #this triangle and the current rect
        #(just one rect)
        #get rect in terms of its points as well (in the 0-32 scale)

        triangle_pos = self.get_points_upscaled()
        intersections = []

        #get_rect_sides gives the sides in the same scale
        #degree as the rectangle passed

        #so need an upscaled rect here given the upscaled triangle
        for r_side in rect.get_sides():
            #print("r: " + str(r_side))
            for t_side in self.get_triangle_sides(True):
                #print("t: " + str(t_side))
                a = r_side[0]
                b = r_side[1]
                c = t_side[0]
                d = t_side[1]
                found_test = get_intersection(a,b,c,d)
                if found_test != None:
                    #print("FOUND: " + str(found_test))
                    intersections.append(found_test)
        #----------------------------------------------------
        #Here we have all the intersections given a single triangle and rect
        for i in intersections:
            print(i)
            #need some sort of check to determine if the intersecting 
            #points are on any of the segments to remove them
            #the extra intersections will mess up the 
            #triangle/rectangle intersection type class
        
            #1. remove single point intersections
            #(might not need to since the vals are floats
            # very rare error to have i think)

            #2. classify into num of intersections 
            #For all, use formulas in notebook and check within the 
            #identified case to set up the points referenced correctly.
            #I think the abs values might let it work regardless
            #but need to look into it for sure
            if len(intersections) == 0:
                #either 0a, 0b
                print()



                #if 0a then triangle is inside rect and area is equal to triangle area
                #
                #point p1 should be the point closest to (0,0)
                #scale all points by the amount it takes to scale p1 to (0,0)
                #
                #now with these scaled points we do the following

                #1. p1_angle = inv_cos((dot_product(p2_s,p3_s)/(magnitude(p2_s) * magnitude(p3_s)
                #2. height = tan(p1_scaled) * magniutude(p3_s)
                #3. area = 1/2 * magnitude(p3_s) * height
                

                #if 0b then rect is inside triangle and area is equal to rect area
                #
                #area = |left point x - right points x} * |top point y - bot point y|




                #STILL GOTTA MAKE SCALED POINTS
                t_points = self.get_points_upscaled()
                r_points = rect.get_points_upscaled()
                closest_to_origin_t = None
                remaining_points = []
                for t in t_points:
                    if closest_to_origin_t == None:
                        closest_to_origin_t = t
                    elif t[0] + t[1] < closest_to_origin_t[0] + closest_to_origin_t[1]:
                        remaining_points.append(closest_to_origin_t)
                        closest_to_origin_t = t
                
                x_diff = closest_to_origin_t[0]
                y_diff = closest_to_origin_t[1]
                
                p2_s = Vector2(remaining_points[0][0],remaining_points[0][1])
                p3_s = Vector2(remaining_points[1][0],remaining_points[1][1])

                p1_angle = math.acos((Vector2.dot_product(p2_s,p3_s))/(p2_s.magnitude * p3_s.magnitude))
                height = math.tan(p1_angle) * p3_s.magnitude
                triangle_area = 1/2 * p3_s.magnitude * height

                print(triangle_area)
                rect_area = 0
                r_width = abs(r_points[0] - r_points[1])
                r_height = abs(r_points[0][0] - r_points[1][0])
                if r_width == 0:
                    r_width = abs(r_points[0][0] - r_points[1][0])
                
                #now test if triangle inside, if rect inside, or if no overlap at all
                if self.isAnyPointInsideRect(rect):
                    return triangle_area
                else if rect.isAnyPointInsideTriangle(self):
                    return rect_area
                else:
                    return 0


            elif len(intersections) == 2:
                #either 2a, 2b, 2c
                #2a is opposite intersections
                print()
            elif len(intersections) == 4:
                #either 4a, 4b
                print()
            elif len(intersections) == 6:
                #6a
                print()

            #3. classify within num of intersections
            


        return

    #get points (0-1)
    def get_points(self):
        points = []
        points.append([self.x1,self.y1])
        points.append([self.x2,self.y2])
        points.append([self.x3,self.y3])
        return points

    #get points(0 - 32)
    def get_points_upscaled(self):
        points = self.get_points()
        count = 0
        for point in points:
            points[count][0] = points[count][0] * 32
            points[count][1] = points[count][1] * 32
            count = count + 1
        return points

    #need the latter for area intersection
    #[TESTED!]
    def get_sides(self,upscaled_bool):
        if upscaled_bool:
            sides = []
            points = self.get_points_upscaled()
            for i in range(len(points)):
                try:
                    sides.append([Vector2(points[i][0],points[i][1]),Vector2(points[i+1][0],points[i+1][1])])
                except:
                    sides.append([Vector2(points[i][0],points[i][1]),Vector2(points[0][0],points[0][1])])

            return sides
        else:
            sides = []
            points = self.get_points()
            for i in range(len(points)):
                try:
                    sides.append([Vector2(points[i][0],points[i][1]),Vector2(points[i+1][0],points[i+1][1])])
                except:
                    sides.append([Vector2(points[i][0],points[i][1]),Vector2(points[0][0],points[0][1])])

            return sides

    def area(x1, y1, x2, y2, x3, y3): 
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

class Rectangle():
    #opacity is handled by triangles, the 3 color values here should be scaled 
    #so the larges value is 255 and the others should be scaled down by the same factor
    def __init__(self,p1,p2,p3,p4,expanded):
        if not expanded:
            self.x1 = p1[0]
            self.y1 = p1[1]

            self.x2 = p2[0]
            self.y2 = p2[1]

            self.x3 = p3[0]
            self.y3 = p3[1]

            self.x4 = p4[0]
            self.y4 = p4[1]

            #all white at first [here white being 0,0,0]
            self.r: float = 0.0
            self.g: float = 0.0
            self.b: float = 0.0

            #determined after color scaling for a model run is decided
            self.greyscale: float
        else:
            self.x1 = p1[0]/32
            self.y1 = p1[1]/32

            self.x2 = p2[0]/32
            self.y2 = p2[1]/32

            self.x3 = p3[0]/32
            self.y3 = p3[1]/32

            self.x4 = p4[0]/32
            self.y4 = p4[1]/32

            self.r: float = 0.0
            self.g: float = 0.0
            self.b: float = 0.0

            #determined after color scaling for a model run is decided
            self.greyscale: float

        return

    #Takes input rect in form of list of 4 lists each describing a points (x,y) w floats
    #Returns the sides of the given rectangle as a list of 2 vector2s
    def get_sides(self,upscaled):
        sides = []

        if upscaled:
            points = self.get_points_upscaled()
        else:
            points = self.get_points()

        sides.append([Vector2(points[0][0],points[0][1]),Vector2(points[1][0],points[1][1])])
        sides.append([Vector2(points[1][0],points[1][1]),Vector2(points[2][0],points[2][1])])
        sides.append([Vector2(points[2][0],points[2][1]),Vector2(points[3][0],points[3][1])])
        sides.append([Vector2(points[3][0],points[3][1]),Vector2(points[0][0],points[0][1])])

        return sides

    def get_points(self):
        return [[self.x1,self.y1],[self.x2,self.y2],[self.x3,self.y3],[self.x4,self.y4]]

    def get_points_upscaled(self):
        return [[self.x1*32,self.y1*32],[self.x2*32,self.y2*32],[self.x3*32,self.y3*32],[self.x4*32,self.y4*32]]

    def update_color(self,triangle,overlap_area):
        #do i need to make sure all the passed values
        #are the expanded version? prob. havent yet tho
        self.r = self.r + (triangle.r * overlap_area)

    def isAnyPointInsideTriangle(self,triangle):
        for i in self.get_points_upscaled():
            if triangle.isPointInside(i):
                return True
        return False

    def isPointInside(self,point):
        biggest = [0,0] #biggest x, biggest_y
        smallest = [32,32] #smallest x, smallest y
        for i in self.get_points_upscaled():
            if biggest[0] < i[0]:
                biggest[0] = i[0]
            if biggest[1] < i[1]:
                biggest[1] = i[1]
            if smallest[0] > i[0]:
                smallest[0] = i[0]
            if smallest[1] > i[1]:
                smallest[1] = i[1]
        if point[0] > biggest[0] or point[0] < smallest[0]:#if in bounds for x
            return False
        if point[1] > biggest[1] or point[1] < smallest[1]:#if in bounds for y'
            return False

        return True


    def __str__(self):
        s = "Rect at {} with color values [r: {}, g: {}, b: {}]".format([self.x1,self.y1], self.r, self.g, self.b)
        return s



class Image():
    def __init__(self,image):
        self.triangles = []
        self.set_full_image_rects_blank()


    def triangles_in_img_format(self):
        #give the triangle layout as an image [3x32x32]
        #[make another method for grayscale img format too]

        #make a 32x32 white image with 0 values
        x = []
        for i in range(32):
            y = []
            for j in range(32):
                y.append(0.0)
            x.append(0.0)
        print(str(len(x)) + "," + str(len(x[0])))

        #add values from triangles

        #1. check which triangles intersect with each rectangle
        #[fill in]
        for triangle in self.triangles:
            #check which rectangles this triangle intersects with
            intersected_rectangles = []

            #still gotta fill this bit
            #if there is an intersection
            for rect in self.rects:
                for r_side in rect.get_sides():
                    for t_side in triangle.get_sides():
                        print()

            for rect in intersected_rectangles:
                #figure out area of overlap of
                #this triangle and the current rect
                area = triangle.get_area_overlap(rect)

        return x

    def set_full_image_rects_blank(self):
        #left to right being 0-32x
        #up to down being 0-32y
        
        #result[0][0] = 0,0
        #results[1][0] = 0,1

        results = []

        for j in range(32):
            for i in range(32):
                rect = Rectangle([i,j],[i+1,j],[i+1,j+1],[i,j+1],True)
                results.append(rect)

        self.rects = results

        return 





#============================================================

if __name__.__contains__("__main__"):
    print("hello world")

    

    #model =  Recreation_Model()

    ##batch, color channels, input size
    #x = torch.randn(64,3,32,32)

    ##passes basic check
    #print(model(x).shape)

    #randimg = torch.randn(32,32)

    #img = Image(randimg)
    #img.triangles_in_img_format()
    #---------------
    t = Triangle([1.1,1.01],[4.3,1.02],[1.91,5.31],False)
    #print(t.get_points_upscaled())
    
    r = [[0,0],[2,0],[2,2],[0,2]]
    
    print("\n")
    print(t.get_area_overlap(r))


    #print(t.get_points())
    #print(t.get_points_upscaled())
    #---------------
    #print(get_rect_sides([[0,4],[5,4],[0,11],[5,11]]))
    #print()
    #print(Image.get_full_image_rects())

    #print(t.get_triangle_sides())
