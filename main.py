import time
from config import TIME
from utils import main

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f"Total execution time:{round(end - start, 2)} seconds", end="")
    if TIME:
        print("(include communication time)")
    else:
        print("(not include communication time)")
