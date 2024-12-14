import sys, os
sys.path.append("..")

from polyfunction import printPoly_ad, calcPoly

## 전역 변수 선언 부분 ##
px = [7, -4, 0, 5]

## 메인 코드 부분 ##
if __name__ == "__main__":

    pStr = printPoly_ad(px)
    print(pStr)

    xValue = int(input("X 값-->"))
    pxValue = calcPoly(xValue, px)
    print(pxValue)