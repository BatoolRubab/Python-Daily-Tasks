import turtle as turtle_module
import random
# import colorgram
# rgb_colors = []
# colors = colorgram.extract('image.jpg', 30)
# for color in colors:
#     r = color.rgb.r
#     g = color.rgb.g
#     b = color.rgb.b
#     new_color = (r , g ,b)
#     rgb_colors.append(new_color)
#
# print(rgb_colors)

tim = turtle_module.Turtle()
turtle_module.colormode(255)
tim.speed("fastest")
tim.penup()
tim.hideturtle()

color_list =[
    (202, 164, 110),(149, 75, 50),(222, 201, 136),(53, 93, 123), (170, 154, 41), (138, 31, 20),
    (134, 163, 184),(197, 92, 73), (47, 121, 86),(73, 43, 35),(145, 178, 149),(14, 98, 70),
    (232, 176, 165),(160, 142, 158),(54, 45, 50),(101, 75, 77),(183, 205, 171),(36, 60, 74),
    (19, 86, 89),(82, 148, 129),(147, 17, 19),(27, 68, 102),(12, 70, 64),(107, 127, 153),(168, 99, 102),
    (64, 27, 70),(25, 25, 112),(85, 107, 47),(139, 0, 0),(0, 100, 0),
]

tim.setheading(225)
tim.forward(300)
tim.setheading(0)
num_of_dots = 101

for dot_count in range(1, num_of_dots):
    tim.dot(20,random.choice(color_list))
    tim.forward(50)

    if dot_count % 10 == 0:
        tim.setheading(90)
        tim.forward(50)
        tim.setheading(180)
        tim.forward(500)
        tim.setheading(0)

screen = turtle_module.Screen()
screen.exitonclick()


