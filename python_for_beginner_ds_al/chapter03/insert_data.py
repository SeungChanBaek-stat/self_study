def insert_data(katok : list, position : int, friend : str):

    if position < 0 or position > len(katok):
        print("데이터를 삽입할 범위를 벗어났습니다.")
        return
    
    katok.append(None)
    kLen = len(katok)

    for i in range(kLen-1, position, -1):
        katok[i] = katok[i-1]
        katok[i-1] = None

    katok[position] = friend

    return katok