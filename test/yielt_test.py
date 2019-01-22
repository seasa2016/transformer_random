def qq():
    arr = []

    for  i in range(10):
        yield i
        print("q")
        arr.append(i)
    
    print(arr)
    yield 111

for i in qq():
    print(i)
else:
    print('a')