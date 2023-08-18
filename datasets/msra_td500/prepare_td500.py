from pathlib import Path
import math

class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


class Rectangle:
    def __init__(self, x, y, w, h, angle):
        # Center Point
        self.x = x
        self.y = y
        # Height and Width
        self.w = w
        self.h = h
        self.angle = angle

    def rotate_rectangle(self, theta):
        pt0, pt1, pt2, pt3 = self.get_vertices_points()

        # Point 0
        rotated_x = math.cos(theta) * (pt0.x - self.x) - math.sin(theta) * (pt0.y - self.y) + self.x
        rotated_y = math.sin(theta) * (pt0.x - self.x) + math.cos(theta) * (pt0.y - self.y) + self.y
        point_0 = Point(rotated_x, rotated_y)

        # Point 1
        rotated_x = math.cos(theta) * (pt1.x - self.x) - math.sin(theta) * (pt1.y - self.y) + self.x
        rotated_y = math.sin(theta) * (pt1.x - self.x) + math.cos(theta) * (pt1.y - self.y) + self.y
        point_1 = Point(rotated_x, rotated_y)

        # Point 2
        rotated_x = math.cos(theta) * (pt2.x - self.x) - math.sin(theta) * (pt2.y - self.y) + self.x
        rotated_y = math.sin(theta) * (pt2.x - self.x) + math.cos(theta) * (pt2.y - self.y) + self.y
        point_2 = Point(rotated_x, rotated_y)

        # Point 3
        rotated_x = math.cos(theta) * (pt3.x - self.x) - math.sin(theta) * (pt3.y - self.y) + self.x
        rotated_y = math.sin(theta) * (pt3.x - self.x) + math.cos(theta) * (pt3.y - self.y) + self.y
        point_3 = Point(rotated_x, rotated_y)
        return point_0, point_1, point_2, point_3


rec = Rectangle(x=x, y=y, w=width, h=height, angle=angle)
rec.rotate_rectangle(angle)


img_path = "/Users/jongbeomkim/Downloads/MSRA-TD500/train/IMG_0030.JPG"
img = load_image(img_path)
show_image(img)

label_path = "/Users/jongbeomkim/Downloads/MSRA-TD500/train/IMG_0030.gt"
for line in open(label_path).readlines():
    idx, difficulty, x, y, width, height, angle = line.split(" ")
