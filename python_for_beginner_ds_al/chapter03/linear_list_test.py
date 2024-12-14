import sys, os
sys.path.append('..')
from chapter03.linear_list import LinearList

## 전역 변수 선언 부분 ##
katok = LinearList([], 0)
select = -1


## 메인 코드 부분 ##
if __name__ == "__main__":
    while (select != 4) :

        select = int(input("선택하세요(1: 추가, 2: 삽입, 3: 삭제, 4: 종료)--> "))

        if (select == 1):
            data = input("추가할 데이터--> ")
            katok_list = katok.add_data(data)
            print(katok_list)
        elif (select == 2):
            pos = int(input("삽입할 위치--> "))
            data = input("추가할 데이터--> ")
            katok_list = katok.insert_data(pos, data)
            print(katok_list)
        elif (select == 3):
            pos = int(input("삽입할 위치--> "))
            katok_list = katok.delete_data(pos)
            print(katok_list)
        elif (select == 4):
            print(katok_list)
            exit
        else:
            print("1~4 중 하나를 입력하세요.")
            continue
