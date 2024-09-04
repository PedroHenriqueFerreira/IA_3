from time import time

def timer(func):
    def inner():
        start = time()
        func()
        end = time()
        print(f'TIME: {end - start}')

        return end - start
        
    return inner

@timer
def aa():
    for i in range(100):
        print(i)
        
print('AA', aa())