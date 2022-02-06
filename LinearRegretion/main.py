class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Loss:
    def MAE(numbers: list):
        # Mean Absolute Error
        return sum(numbers)/len(numbers)

    def MSE(numbers: list):
        # Mean Squared Error
        return 0.5 * (sum(x*x for x in numbers)/len(numbers))

class LinearRegression:
    # In form of y = mx + b
    def __init__(self, alpha = 0.1, m = 1, b = 0):
        self.alpha = alpha # learning trate
        self.m = m # slope
        self.b = b # Vertical shift

    def show(self):
        # Print function
        print(f'y = {round(self.m,2)}x + {round(self.b,2)}')

    def solve(self, x):
        # Solve function for 'y'
        return self.m * x + self.b

    def absolute(self, point):
        # Absolute trick
        m = point.x * self.alpha
        b = self.alpha
        # Update values
        if (point.y - self.solve(point.x)) > 0:
            self.m += m
            self.b += b
        else: 
            self.m -= m
            self.b -= b

    def square(self, point):
        # Square trick
        m = point.x * (point.y - self.solve(point.x)) * self.alpha
        b = (point.y - self.solve(point.x)) * self.alpha
        # Update values
        self.m += m
        self.b += b

    def mae(self, points):
        return Loss.MAE( list( abs(self.solve(p.x)-p.y) for p in points) )

    def mse(self, points):
        return Loss.MSE( list( abs(self.solve(p.x)-p.y) for p in points) )