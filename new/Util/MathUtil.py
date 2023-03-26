from decimal import Decimal


class MathUtil:

    @staticmethod
    def round(*numbers, decimal=0):
        ret = []
        try:
            for _ in numbers:
                ret.append(round(_, decimal))
        except TypeError as e:
            print('MathUtil.round... TypeError: {}'.format(e))
            return []
        return ret


    @staticmethod
    def add(left, right):
        return float(Decimal(str(left)) + Decimal(str(right)))


    @staticmethod
    def sub(left, right):
        return float(Decimal(str(left)) - Decimal(str(right)))


if __name__ == '__main__':
    # print(MathUtil.round(1,2.2123,'123',4))
    MathUtil.add(3.2, 1)
    pass
