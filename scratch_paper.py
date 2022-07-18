from turtle import Vec2D
from Vector2 import Vector2 


a = Vector2(1,1)
b = Vector2(3,2)
c = Vector2(0,3)
d= Vector2(2,1)

def get_intersection(a,b,c,d):
    ab = b - a
    cd = d - c

    ab_cross_cd = ab.cross(cd)

    if ab_cross_cd == 0:
        return None
    else:
        ac = c - a
        t1 = ac.cross(cd) / ab_cross_cd
        t2 = -ab.cross(ac) / ab_cross_cd
        return a + ab.scaled(t1), t1, t2

if __name__ == "__main__":
    print(get_intersection(a,b,c,d))