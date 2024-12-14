import sys, os
sys.path.append('..')

def delete_data(katok:list, position:int):
    if position < 0 or position >= len(katok):
        print("데이터를 삭제할 범위를 벗어났습니다.")
        return

    kLen = len(katok)
    katok[position] = None

    for i in range(position+1, kLen):
        katok[i-1] = katok[i]
        katok[i] = None

    del(katok[kLen-1])

    return katok



def delete_all_data(katok:list, position:int):  
    if position < 0 or position >= len(katok):
        print("데이터를 삭제할 범위를 벗어났습니다.")
        return
    
    katok[position] = None

    while katok[-1] is not None:
        del(katok[-1])

    del(katok[-1])
    
    return katok