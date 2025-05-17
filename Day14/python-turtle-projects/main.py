import turtle
from turtle import Turtle , Screen
import random
tim = Turtle()
tim.shape("turtle")
tim.color("DarkGreen")
tim.speed("fastest")
screen = turtle.Screen()
screen.bgcolor("black")


turtle.colormode(255)
def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    r_color = (r , g , b)
    return r_color
# TODO:1  draw square on screen
# for _ in range(4):
#     tim.forward(100)
#     tim.right(90)
# TODO: 2 draw dash line
# for _ in range(15):
#     tim.forward(10)
#     tim.penup()
#     tim.forward(10)
#     tim.pendown()
# TODO: 3 drawing different shapes form 3 sides to 10 sided shape and every shape boundary has different color
# def draw_shapes(num_sides):
#     angle = 360 / num_sides
#     for _ in range(num_sides):
#         tim.color(random.choice(random_color()))
#         tim.forward(100)
#         tim.right(angle)
#
# for shape in range(3,11):
#     draw_shapes(shape)

#  TODO: 4 Random walk on screen using different colors , also change the thickness of pen

# directions = [0,90,180,270]
# tim.pensize(10)
# tim.speed("fastest")
# for _ in range(200):
#     tim.forward(20)
#     tim.color(random_color())
#     tim.setheading(random.choice(directions))

# TODO: 5 make a spirograph

def draw_spirograph(size_of_gap):
    for _ in range(int(360 / size_of_gap)):
        tim.color(random_color())
        tim.circle(100)
        current_heading = tim.heading()
        tim.setheading(tim.heading() + size_of_gap)

draw_spirograph(5)

screen = Screen()
screen.exitonclick()
