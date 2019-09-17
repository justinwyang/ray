import time
import ray

ray.init()

@ray.remote
def foo():
    raise Exception("oops")

@ray.remote
def main():
    i = 0
    while True:
        i += 1
        time.sleep(1)
        print(i)
        try:
            ray.get(foo.remote())
        except:
            pass

ray.get(main.remote())
