# name = input("Please Enter your name ")
# def greet():
#     print(f"Hello {name}")
#     print(f"Have a good day {name}")
# greet()

# function with inputs
def greet_with_name(name):
    print(f"Hello {name}")
    print(f"Have a good day {name}")
greet_with_name("batool")
# functions with more than 1 inputs , calling positional arguments

def greet_with(name , location):
    print(f"Hello {name}")
    print(f"What is it like in {location}")
# greet_with("Rubab","Islamabad")

# calling with keyword arguments
greet_with(location="Islamabad",name="Rubab")
