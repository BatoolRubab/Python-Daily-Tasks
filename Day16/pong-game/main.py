from turtle import Turtle , Screen
from paddle import Paddle
from ball import Ball
from scoreboard import ScoreBoard
import time


screen = Screen()
screen.bgcolor("Black")
screen.setup(height= 600, width= 800)
screen.title("Pong 🏓")
screen.tracer(0)

center_line = Turtle()
center_line.color("white")
center_line.penup()
center_line.goto(0, 300)
center_line.setheading(270)
for _ in range(30):
    center_line.pendown()
    center_line.forward(10)
    center_line.penup()
    center_line.forward(10)
center_line.hideturtle()


r_paddle = Paddle((350 , 0))
l_paddle = Paddle((-350 , 0))
ball = Ball()
score_board = ScoreBoard()


screen.listen()
screen.onkey(r_paddle.go_up,"Up")
screen.onkey(r_paddle.go_down,"Down")
screen.onkey(l_paddle.go_up,"u")
screen.onkey(l_paddle.go_down,"d")



game_is_on = True
while game_is_on:
    time.sleep(ball.move_speed)
    screen.update()
    ball.move()

    #detect collision with wall
    if ball.ycor() > 290 or ball.ycor() < -290:
        # need to bounce back
        ball.bounce_y()

    #detech collision with paddle
    if ball.distance(r_paddle) < 50 and ball.xcor() > 320 or ball.distance(l_paddle) < 50 and ball.distance(l_paddle) > -320:
        ball.bounce_x()

    # detect R paddle misses
    if ball.xcor() > 380:
        ball.reset_position()
        score_board.l_point()

    # detect L paddle misses
    if ball.xcor() < -380:
        ball.reset_position()
        score_board.r_point()

    # check for game over
    if score_board.l_score >= 10:
        score_board.game_over("Left Player")
        game_is_on = False
    elif score_board.r_score >= 10:
        score_board.game_over("Right Player")
        game_is_on = False

screen.exitonclick()

