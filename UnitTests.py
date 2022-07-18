import unittest
from Vector2 import Vector2 

from IRM import Rectangle, Triangle, are_side_groups_equal


class UnitTests(unittest.TestCase):

    def testing_test(self):
        self.assertTrue(True)

    def test_triangle_get_triangle_sides(self):
        t = Triangle([.2,.3],[.99,.22],[.34,.76],False)

        correct_sides = [[Vector2(.2,.3),Vector2(.99,.22)],[Vector2(.99,.22),Vector2(.34,.76)],[Vector2(.34,.76),Vector2(.2,.3)]]
        correct_sides_upscaled = [[Vector2(.2*32,.3*32),Vector2(.99*32,.22*32)],[Vector2(.99*32,.22*32),Vector2(.34*32,.76*32)],[Vector2(.34*32,.76*32),Vector2(.2*32,.3*32)]]

        #get_sides(True) gives upscaled vals while False gives 0-1 values
        self.assertTrue(t.get_sides(True) == correct_sides_upscaled)
        self.assertTrue(t.get_sides(False) == correct_sides)

    def test_get_rect_sides(self):
        #r = Rectangle([.2,.3],[.99,.3],[.2,.76],[.99,76],False)
        r = Rectangle([.2,.3],[.99,.3],[.2,.76],[.99,.76],False)

        #correct_sides = [[Vector2(.2,.3),Vector2(.99,.3)],[Vector2(.99,.3),Vector2(.2,.76)],[Vector2(.2,.76),Vector2(.99,.76)],[Vector2(.99,.76),Vector2(.2,.3)]]
        correct_sides = [[Vector2(.2,.3),Vector2(.99,.3)],[Vector2(.99,.3),Vector2(.2,.76)],[Vector2(.2,.76),Vector2(.99,.76)],[Vector2(.99,.76),Vector2(.2,.3)]]
        correct_sides_upscaled = [[Vector2(.2*32,.3*32),Vector2(.99*32,.3*32)],[Vector2(.99*32,.3*32),Vector2(.2*32,.76*32)],[Vector2(.2*32,.76*32),Vector2(.99*32,.76*32)],[Vector2(.99*32,.76*32),Vector2(.2*32,.3*32)]]

        #self.assertTrue(r.get_sides(True) == correct_sides)
        self.assertTrue((are_side_groups_equal(r.get_sides(False),correct_sides)))
        #self.assertTrue(r.get_sides(False) == correct_sides_upscaled)
        self.assertTrue(are_side_groups_equal(r.get_sides(True),correct_sides_upscaled))

if __name__ == "__main__":
    #t = Triangle([.2,.3],[.99,.22],[.34,.76],False)
    r = Rectangle([.2,.3],[.99,.3],[.2,.76],[.99,.76],False)
    #print(r.get_sides(True))
    #print(r.get_points())
    #print(r.get_sides(False))
    #print([[Vector2(.2*32,.3*32),Vector2(.99*32,.3*32)],[Vector2(.99*32,.3*32),Vector2(.99*32,.76*32)],[Vector2(.99*32,.76*32),Vector2(.3*32,.76*32)],[Vector2(.3*32,.76*32),Vector2(.2*32,.3*32)]])
    #print([[Vector2(.2,.3),Vector2(.99,.3)],[Vector2(.99,.3),Vector2(.2,.76)],[Vector2(.2,.76),Vector2(.99,.76)],[Vector2(.99,.76),Vector2(.2,.3)]])
    correct_sides = [[Vector2(.2,.3),Vector2(.99,.3)],[Vector2(.99,.3),Vector2(.2,.76)],[Vector2(.2,.76),Vector2(.99,.76)],[Vector2(.99,.76),Vector2(.2,.3)]]
    print(are_side_groups_equal(r.get_sides(False),correct_sides))
    

    unittest.main()