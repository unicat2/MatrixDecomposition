
def qr_decomposition(A, b, method):
    """
    QR_Decomposition - 计算矩阵A的QR分解并求解Ax=b
    
    parameters:
    A - 待分解的矩阵
    b - 方程组Ax=b的右端向量
    method - 选择QR分解的方法:
             'householder' - 使用Householder变换
             'givens'      - 使用Givens旋转
             'mgs'         - 使用改进的Gram-Schmidt过程
    
    returns:
    Q - 正交矩阵
    R - 上三角矩阵
    x - 方程组Ax=b的解
    """
    m, n = len(A), len(A[0])

    if method == 'householder':
        Q, R = householder_qr(A)
    elif method == 'givens':
        Q, R = givens_qr(A)
    elif method == 'mgs':
        Q, R = mgs_qr(A)
    else:
        raise ValueError("请选择'householder'、'givens'或'mgs'。")

    # 求解Ax = b
    if b is not None:
        if m >= n:  # 高或方阵
            y = matrix_vector_multiply(matrix_transpose(Q), b) # 计算Q^Tb
            if any(R[i][i] == 0 for i in range(n)):  # 检查R的对角线元素
                if all(v == 0 for v in b): # b全为0
                    x = "无穷多解"
                else: # b不全为0, 但R的对角线元素有0
                    x = "无解"
            else: # R的对角线元素均不为0
                x = back_substitution(R, y[:n])  # 只取前n个元素
        else:  # 宽矩阵
            x = least_squares_solution(Q, R, b)
    else:
        x = None

    return Q, R, x

def least_squares_solution(Q, R, b):
    """
    使用QR分解计算最小二乘解
    """
    Qb = matrix_vector_multiply(matrix_transpose(Q), b) # 计算Q^Tb
    
    return back_substitution_with_tolerance(R, Qb) # 只对非零对角元素进行回代

def back_substitution_with_tolerance(R, y, tol=1e-10):
    """
    回代求解方程组Rx=y，忽略小于给定阈值的对角元素
    与back_substitution的区别是，对于上三角矩阵 (R) 的对角元素，如果它绝对值小于给定的容差tol，则认为是0，避免数值不稳定性
    宽矩阵的情况下，只对前m个元素进行回代
    """
    n = len(y)
    x = [0] * n
    for i in range(n-1, -1, -1):
        if abs(R[i][i]) > tol:  # 忽略小于给定阈值的对角元素
            x[i] = (y[i] - sum(R[i][j] * x[j] for j in range(i+1, n))) / R[i][i]
    return x


def back_substitution(R, y):
    """
    回代求解方程组
    """
    n = len(y)
    x = [0] * n
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(R[i][j] * x[j] for j in range(i+1, n))) / R[i][i]
    return x

def forward_substitution(R, y):
    """
    前代法求解方程组
    """
    n = len(y)
    x = [0] * n
    for i in range(n):
        x[i] = (y[i] - sum(R[i][j] * x[j] for j in range(i))) / R[i][i]
    return x




def householder_qr(A):
    """
    Householder_QR - 使用Householder变换计算矩阵A的QR分解
    计算原理:
    1. 对于第k列，计算Householder向量v，使得Hv = [vector_norm(x), 0, ..., 0]
    2. 计算Householder矩阵H = I - 2vv^T
    3. 计算R = HA，Q = H1H2...Hn

    """
    m = len(A) 
    n = len(A[0]) 
    Q = identity_matrix(m) 
    R = [row[:] for row in A] 
    
    for k in range(min(m, n)): 
        x = [R[i][k] for i in range(k, m)] # 每一列作为向量x   
        if vector_norm(x) == 0:
            continue # 如果x的范数为0，跳过
        e1 = [1] + [0] * (len(x) - 1) # e1 = [1, 0, ..., 0]
        v = vector_add(scalar_vector_multiply(sign(x[0]) * vector_norm(x), e1), x) # v = sign(x[0])||x||e1 + x
        v = scalar_vector_multiply(1/vector_norm(v), v) # v = v/||v||
        H = householder_matrix(v, m, k) # 计算Householder矩阵, H = I - 2vv^T
        R = matrix_multiply(H, R) # R = HR
        Q = matrix_multiply(Q, matrix_transpose(H)) # Q = QH^T
    
    return Q, R

def householder_matrix(v, m, k):
    """
    计算Householder矩阵H = I - 2vv^T

    parameters:
    v - Householder向量
    m - 矩阵A的行数
    k - 第k列

    returns:
    H - Householder矩阵
    """
    H = identity_matrix(m)
    for i in range(k, m):
        for j in range(k, m):
            H[i][j] -= 2 * v[i-k] * v[j-k]
    return H

def givens_qr(A):
    """
    计算矩阵A的QR分解，使用Givens reduction
    
    parameters:
    A - 输入矩阵，大小为m x n，其中m是行数，n是列数。

    returns:
    Q - 正交矩阵，大小为m x m。
    R - 上三角矩阵，大小为m x n。
    """

    m = len(A)  
    n = len(A[0])  
    Q = identity_matrix(m) 
    R = [row[:] for row in A] 
    
    for j in range(n):
        for i in range(m-1, j, -1):
            G = identity_matrix(m)  
            # 计算Givens旋转系数c和s，使得R[i][j]变为0
            c, s = givens_rotation(R[i-1][j], R[i][j])
            G[i-1][i-1] = c 
            G[i][i] = c  
            G[i-1][i] = s  
            G[i][i-1] = -s 
            R = matrix_multiply(G, R)  # R = GR
            Q = matrix_multiply(Q, matrix_transpose(G))  # Q = QG^T
    return Q, R  


def givens_rotation(a, b):
    """
    Givens_Rotation - 计算Givens旋转矩阵
    计算公式：
    c = a / (a^2 + b^2)^0.5
    s = b / (a^2 + b^2)^0.5

    parameters:
    a, b - 两个对应位置的元素
    returns:
    c, s - cos(theta), sin(theta)
    """
    if b == 0:
        c = 1
        s = 0
    else:
        if abs(b) > abs(a):
            r = a / b
            s = 1 / (1 + r**2)**0.5 
            c = s * r
        else:
            r = b / a
            c = 1 / (1 + r**2)**0.5
            s = c * r
    return c, s


def mgs_qr(A):
    """
    计算矩阵A的QR分解，使用改进的Gram-Schmidt正交化方法。
    parameters:
    A - 输入矩阵，大小为m x n，其中m是行数，n是列数。
    returns:
    Q - 正交矩阵，大小为m x n。
    R - 上三角矩阵，大小为n x n。
    """

    m = len(A)  
    n = len(A[0])  
    Q = [[0] * n for _ in range(m)]  # m x n
    R = [[0] * n for _ in range(n)]  # n x n
    
    # 对于每一列k，进行Gram-Schmidt正交化
    for k in range(n):
        # 计算R[k][k]，即A的第k列的范数
        R[k][k] = vector_norm([A[i][k] for i in range(m)])
        # 如果R[k][k]为0，表示A的第k列是零向量，跳过此列
        if R[k][k] == 0:
            continue
        # 对第k列进行归一化，Q[:,k] = A[:,k] / R[k][k]
        Q_col = scalar_vector_multiply(1/R[k][k], [A[i][k] for i in range(m)])
        for i in range(m):
            Q[i][k] = Q_col[i]  
        # 对于每一列j从k+1到n，计算R[k][j]并更新A的第j列
        for j in range(k+1, n):
            # 计算R[k][j] = Q[:,k]^T * A[:,j]，等于Q的第k列与A的第j列的内积
            R[k][j] = vector_dot(Q_col, [A[i][j] for i in range(m)]) 
            # 更新A的第j列，A[:,j] = A[:,j] - Q[:,k] * R[k][j]，j从k+1到n都要更新
            A_col = vector_subtract([A[i][j] for i in range(m)], scalar_vector_multiply(R[k][j], Q_col))
            for i in range(m):
                A[i][j] = A_col[i]  # 更新A的第j列
    return Q, R  



def vector_norm(v):
    """
    向量范数
    """
    return sum(x**2 for x in v) ** 0.5

def vector_dot(v, w):
    """
    向量点积
    """
    return sum(v[i] * w[i] for i in range(len(v)))

def vector_add(v, w):
    """
    向量加法
    """
    return [v[i] + w[i] for i in range(len(v))]

def vector_subtract(v, w):
    """
    向量减法
    """
    return [v[i] - w[i] for i in range(len(v))]

def scalar_vector_multiply(scalar, v):
    """
    标量乘向量
    """
    return [scalar * x for x in v]

def identity_matrix(size):
    """
    大小为size的单位矩阵
    """
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

def matrix_multiply(A, B):
    """
    矩阵乘法
    """
    m, n = len(A), len(B[0])
    result = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(len(B)))
    return result

def matrix_transpose(A):
    """
    矩阵转置
    """
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def matrix_vector_multiply(A, v):
    """
    矩阵乘向量
    """
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def sign(x):
    return 1 if x >= 0 else -1
