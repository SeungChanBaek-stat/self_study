import sys, os
sys.path.append("..")

from polyfunction import printPoly_up, calcPoly_up

## 전역 변수 선언 부분 ##
tx = [300, 20, 0]
px = [7, -4, 5]

## 메인 코드 부분 ##
if __name__ == "__main__":

    pStr = printPoly_up(tx,px)
    print(pStr)

    xValue = int(input("X 값-->"))
    pxValue = calcPoly_up(xValue, tx, px)
    print(pxValue)