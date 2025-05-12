from art import logo
print(logo)

# Define operation functions
def add(n1, n2):
    return n1 + n2

def subtract(n1, n2):
    return n1 - n2

def multiply(n1, n2):
    return n1 * n2

def divide(n1, n2):
    if n2 == 0:
        return "Infinity"
    return n1 / n2

# Dictionary of operations
operations = {
    "+": add,
    "-": subtract,
    "*": multiply,
    "/": divide,
}

def calculator():
    num1 = float(input("What's the first number?: "))

    while True:
        for symbol in operations:
            print(symbol)

        operation_symbol = input("Pick an operation: ").strip()

        if operation_symbol not in operations:
            print("Invalid operation. Please try again.")
            continue

        num2 = float(input("What's the next number?: "))

        # Check if previous result was 'Infinity'
        if isinstance(num1, str) and num1 == "Infinity":
            result = "Infinity"
        else:
            result = operations[operation_symbol](num1, num2)

        print(f"{num1} {operation_symbol} {num2} = {result}")

        choice = input("Type 'y' to continue with the result, 'n' to start new, or 'exit' to quit: ").lower().strip()

        if choice == "y":
            num1 = result if result == "Infinity" else float(result)
        elif choice == "n":
            print("\n" * 3)
            calculator()  # restart the calculator
            return
        elif choice == "exit":
            print("Goodbye! 👋")
            return
        else:
            print("Invalid input. Exiting.")
            return

# Start calculator
calculator()




