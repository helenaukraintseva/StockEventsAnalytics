a = {"x": 1, "z": 2, "y": 3, "d": 5, "e": 56}


def func(x, y, z, g=6, **d):
    print("x:", x)
    print("y:", y)
    print("z:", z)
    print(d)
    print(g)
    print("OK")


func(**a)