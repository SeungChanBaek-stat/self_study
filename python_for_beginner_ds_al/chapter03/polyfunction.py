def printPoly(p_x):
    term = len(p_x) - 1     # 최고차항 숫자 = 배열 길이 - 1
    polyStr = "P(x) = "

    for i in range(len(p_x)):
        coef = p_x[i]   # 계수

        if (coef >= 0):
            polyStr += "+"
        polyStr += str(coef) + "x^" + str(term) + " "
        term -= 1

    return polyStr



def calcPoly(xVal, p_x):
    retValue = 0
    term = len(p_x) - 1

    for i in range(len(p_x)):
        coef = p_x[i]
        retValue += coef * xVal ** term
        term -= 1

    return retValue



def printPoly_ad(p_x):
    term = len(p_x) - 1     # 최고차항 숫자 = 배열 길이 - 1
    polyStr = "P(x) = "

    for i in range(len(p_x)):
        coef = p_x[i]   # 계수

        if (i == 0) & (coef > 0):
            polyStr += str(coef) + "x^" + str(term) + " "
        elif (coef > 0):
            polyStr += "+" + str(coef) + "x^" + str(term) + " "
        elif (coef == 0):
            term -= 1
            continue
        else: 
            polyStr += str(coef) + "x^" + str(term) + " "
        term -= 1

    return polyStr




def printPoly_up(t_x, p_x):
    polyStr = "P(x) = "

    for i in range(len(p_x)):
        term = t_x[i]
        coef = p_x[i]   # 계수

        if (i == 0) & (coef > 0):
            polyStr += str(coef) + "x^" + str(term) + " "
        elif (coef > 0):
            polyStr += "+" + str(coef) + "x^" + str(term) + " "
        elif (coef == 0):
            term -= 1
            continue
        else: 
            polyStr += str(coef) + "x^" + str(term) + " "

    return polyStr



def calcPoly_up(xVal, t_x, p_x):
    retValue = 0

    for i in range(len(p_x)):
        term = t_x[i]
        coef = p_x[i]
        retValue += coef * xVal ** term

    return retValue