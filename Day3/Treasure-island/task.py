print(r'''
*******************************************************************************
          |                   |                  |                     |
 _________|________________.=""_;=.______________|_____________________|_______
|                   |  ,-"_,=""     `"=.|                  |
|___________________|__"=._o`"-._        `"=.______________|___________________
          |                `"=._o`"=._      _`"=._                     |
 _________|_____________________:=._o "=._."_.-="'"=.__________________|_______
|                   |    __.--" , ; `"=._o." ,-"""-._ ".   |
|___________________|_._"  ,. .` ` `` ,  `"-._"-._   ". '__|___________________
          |           |o`"=._` , "` `; .". ,  "-._"-._; ;              |
 _________|___________| ;`-.o`"=._; ." ` '`."\ ` . "-._ /_______________|_______
|                   | |o ;    `"-.o`"=._``  '` " ,__.--o;   |
|___________________|_| ;     (#) `-.o `"=.`_.--"_o.-; ;___|___________________
____/______/______/___|o;._    "      `".o|o_.--"    ;o;____/______/______/____
/______/______/______/_"=._o--._        ; | ;        ; ;/______/______/______/_
____/______/______/______/__"=._o--._   ;o|o;     _._;o;____/______/______/____
/______/______/______/______/____"=._o._; | ;_.--"o.--"_/______/______/______/_
____/______/______/______/______/_____"=.o|o_.--""___/______/______/______/____
/______/______/______/______/______/______/______/______/______/______/_____ /
*******************************************************************************
''')
print("Welcome to Treasure Island.")
print("Your mission is to find the treasure.")
choice1 = input('You\'re at cross road. Where you want to go? Type "left" or "right".\n') .lower()

if choice1 == "right":
    choice2= input(' You have come to a lake. '
                   'There is a island in the middle of the late.'
          'type "swim" to swim across  or "wait" to wait for a boat wait\n').lower()
    if choice2 == "wait":
       choice3 = input("You arrived at island."
                       "There is house with 3 doors. red , yellow, blue."
                       " Which color do you choose?\n")
       if choice3 == "red":
           print("game over")
       elif choice3 == "yellow":
           print("game over")
       elif choice3 == "blue":
           print("congratulations! You Win")
       else:
           print("You choose something else. Game over")
    else:
        print("game over")


else:
    print("Game Over")

