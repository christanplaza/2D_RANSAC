import numpy as np

class PolynomialLeastSquare:
    """
    多項式最小二乗法
    
    Attributes
    ----------
    coefficients : int
        使用する弱分類器の数
    d : int
        fitting を行う多項式の次数
    """
    
    def fit(self, x, y, d):
        """
        回帰曲線の係数を計算
        
        Parameters
        ----------
        x : numpy array [float]
        y : numpy array [float]
        d : int
            fitting する多項式の次数
        """
        n = len(x)
        # x のべき乗を計算
        x_pow = [np.full(n, 1.)]
        for i in range(2*d):
            x_pow.append(x_pow[-1] * x)
        x_pow = np.array(x_pow)
        # 行列 S を計算
        s = []
        for i in range(2*d+1):
            s.append(np.sum(x_pow[i]))
        S = np.zeros([d+1, d+1])
        for i in range(d+1):
            for j in range(d+1):
                S[i][j] = s[i+j]
        # ベクトル t を計算
        t = []
        for i in range(d+1):
            t.append(np.sum(x_pow[i]*y))
        t = np.matrix(t).T
        # 係数を求める
        S_inv = np.linalg.pinv(S)
        self.coefficients = np.array(np.dot(S_inv, t)).flatten()
        self.d = d
    
    def predict(self, x):
        """
        学習済み回帰曲線を使って未知の x を変換
        
        Parameters
        ----------
        x : numpy array [float]
        """
        x_pow = np.full(len(x), 1.)
        y = np.zeros(len(x))
        for i in range(len(self.coefficients)):
            y += x_pow * self.coefficients[i]
            x_pow *= x
        return y