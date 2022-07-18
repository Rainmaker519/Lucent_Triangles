from turtle import *;

class Turtle:
    pass

color('red','yellow')
speed(100)
begin_fill()
while True:
    forward(200)
    left(170)
    if abs(pos()) < 1:
        break
end_fill()
done()



#genetic algorithm where turtle tries to recreate an image
#where the way a model is evaluated is by how close in terms of
#overall color+tone the image is to the original (for cmyk and b/w)

#--OR--

#build a turtle software which visualizes neural network architectures
#and updates the colors of the connections as training occurs live
#(prob gonna have to approx a change over the time frame between
# updates)




#just ok! hello world

#iIaA - editing in line

#i and I insert BEFORE the current character or line
#a and A insert AFTER the current character or

#hjkl - navigation

#xr - making changes in command mode

#x - delete current character (in command mode)

#r - replace current character with subsequently
#       inputted character(in command mode)