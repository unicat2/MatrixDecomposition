from function import *

def test_qr_decomposition():
    test_cases = [
        # 方阵
        {"A": [[12, -51, 4], [6, 167, -68], [-4, 24, -41]], "b": [1, 2, 3], "desc": "Square matrix"},
        # 高矩阵
        {"A": [[1, 2], [3, 4], [5, 6]], "b": [7, 8, 9], "desc": "Tall matrix"},
        # 宽矩阵
        {"A": [[1, 2, 3], [4, 5, 6]], "b": [7, 8], "desc": "Wide matrix"},
        # 单位矩阵
        {"A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "b": [1, 2, 3], "desc": "Identity matrix"},
        # 零矩阵
        {"A": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], "b": [0, 0, 0], "desc": "Zero matrix"},
        # 随机矩阵
        {"A": [[3.5, 5.2, 7.1], [2.2, 6.8, 1.4], [4.9, 8.3, 9.6]], "b": [10.1, 11.2, 12.3], "desc": "Random matrix"},
        # 奇异矩阵
        {"A": [[2, 4, 6], [1, 2, 3], [3, 6, 9]], "b": [1, 2, 3], "desc": "Singular matrix"},
        # 对称矩阵
        {"A": [[4, 1, 2], [1, 2, 3], [2, 3, 6]], "b": [1, 2, 3], "desc": "Symmetric matrix"},
        # 上三角矩阵
        {"A": [[1, 2, 3], [0, 4, 5], [0, 0, 6]], "b": [1, 2, 3], "desc": "Upper triangular matrix"},
    ]
   

    failed_cases = []

    for case in test_cases:
        A = case["A"]
        b = case["b"]
        description = case["desc"]

        for method in ['householder', 'givens', 'mgs']:
            print(f'Testing method: {method} on {description}')
            Q, R, x = qr_decomposition(A, b, method)
            qr_correct = verify_qr_decomposition(A, Q, R)
            solution_correct = verify_solution(A, b, x)
            
            if not qr_correct or not solution_correct:
                fail_reason = "QR分解" if not qr_correct else "解方程组"
                failed_cases.append((description, method, fail_reason))
            
            print_results(Q, R, x)

    if failed_cases:
        print("\nFailed test cases!!!:")
        for desc, method, reason in failed_cases:
            print(f"Method: {method} on {desc} failed due to {reason}")
    else:
        print("\nAll test cases passed successfully.")

def verify_qr_decomposition(A, Q, R, tol=1e-3):
    """
    验证QR分解是否正确，即A ≈ QR
    """
    m, n = len(A), len(A[0])
    QR = matrix_multiply(Q, R)
    for i in range(m):
        for j in range(n):
            if abs(QR[i][j] - A[i][j]) > tol:
                return False
    return True

def verify_solution(A, b, x, tol=1e-6):
    """
    验证解x是否满足Ax ≈ b
    """
    if isinstance(x, str):  # 如果x是"无解"或"无穷多解"，跳过验证
        return True
    Ax = matrix_vector_multiply(A, x)
    for i in range(len(b)):
        if abs(Ax[i] - b[i]) > tol:
            return False
    return True

def print_results(Q, R, x):
    print('Q矩阵:')
    for row in Q:
        print(row)
    print('R矩阵:')
    for row in R:
        print(row)
    print('Ax = b 的解 x:')
    print(x)
    print('')


if __name__ == '__main__':
    test_qr_decomposition()