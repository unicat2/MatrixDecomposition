
from function import *



def main():


    # 运行测试用例
    # test_qr_decomposition()

    # 根据输入进行测试
    # 输入矩阵A和向量b的维度
    m = int(input("请输入矩阵A的行数: "))
    n = int(input("请输入矩阵A的列数: "))

    # 输入矩阵A的元素
    A = []
    print("请输入矩阵A的元素:")
    for i in range(m):
        row = []
        for j in range(n):
            element = float(input(f"A[{i+1}][{j+1}]: "))
            row.append(element)
        A.append(row)

    # 输入向量b的元素
    b = []
    print("请输入向量b的元素:")
    for i in range(m):
        element = float(input(f"b[{i+1}]: "))
        b.append(element)
        

    

    # 选择QR分解的方法
    method = input("请选择QR分解的方法('householder', 'givens', 'mgs'): ")

    print("方法: ", method)
    print("A矩阵:")
    for row in A:
        print(row)
    print("b向量:")
    print(b)
    # qr_decomposition函数进行计算
    Q, R, x = qr_decomposition(A, b, method)

    # 输出结果
    print("Q矩阵:")
    for row in Q:
        print(row)
    print("R矩阵:")
    for row in R:
        print(row)
    print("Ax = b 的解 x:")
    print(x)

if __name__ == '__main__':
    while True:
        main()