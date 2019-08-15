import numpy as np
import click
from scipy import optimize

@click.command()
@click.option('--coefficients', default=(1,0,2), help='tuple for representation of polynomial')
def ployvalTest(coefficients: tuple =(1,0,2) ) -> str:
    print('Input: ', coefficients)
    res = np.polyval(coefficients, 1)
    if res == 3:
        return 'descending'
    else:
        return 'ascending'



@click.command()
@click.option('--value', help='value of bond')
@click.option('--price', help='price of bond')
@click.option('--couponRate', help='couponRate of bond')
@click.option('--period', help='period of bond')
# 跟 wind 实盘 最新YTM 对不上 ！！！！！！！！！！
def normalBondYTM(value: float, price: float, couponRate: float, period: int):
    coupon = value * couponRate
    poly = np.array([-price] + [coupon] * (period-1) + [coupon+value])
    roots = np.roots(poly)
    for root in roots:
        if root.imag == 0.0:
            return root.real - 1


# 跟 wind 实盘 久期 对不上 ！！！！！！！！！！
def normalBondDuration(value: float, price: float, couponRate: float, period: int):
    YTM = normalBondYTM(value, price, couponRate, period)
    vec1 = np.array([i*np.exp(-YTM * i) for i in range(period+1)[1:]])
    coupon = value * couponRate
    vec2 = np.array([coupon] * (period -1) + [coupon+value])

    if price != 0:
        continuousDuration = vec1.dot(vec2) / price
        modifiedDuration = continuousDuration/(1+YTM)
        return modifiedDuration
    else:
        print('price is zero')
        return



if __name__ == '__main__':
    # print(ployvalTest())
    # print(normalBondYTM(100, 90.68, 0.0375, 5))
    # print(normalBondDuration(100, 90.68, 0.0375, 5))
    sen = ployvalTest()
    print(sen)