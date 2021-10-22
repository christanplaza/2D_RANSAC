import numpy as np
from PolynomialLeastSquare import PolynomialLeastSquare

class RANSAC:
    """
    Attributes
    ----------
    mse : float
        平均二乗誤差 (Mean Square Error)
    ls_best : object
        最適な最小二乗法モデル
    x_inliers_best : numpy array [float]
        最適モデルを学習した際の正常値 x
    y_inliers_best : numpy array [float]
        最適モデルを学習した際の正常値 y
    """
    
    def __init__(self, max_trials, residual_threshold, min_inliers_rate):
        """
        Parameters
        ----------
        max_trials : int
            ランダムサンプリングによる fitting を試行する最大回数
        residual_threshold : float
            ランダムサンプルから学習したモデルとの残差がこれ以内であれば「正常値」とみなす
        min_inliers_rate : float
            [0, 1] の小数。データサンプル全体に占める正常値の割合がこれ以下のものは最適モデルの候補に含めない
        """
        self.max_trials = max_trials
        self.residual_threshold = residual_threshold
        self.min_inliers_rate = min_inliers_rate
    
    def predict(self, x):
        print(self.ls_best)
        return self.ls_best.predict(x)
    
    def fit(self, x, y, d, selectedIndex):
        """
        Parameters
        ----------
        x : numpy array [float]
        y : numpy array [float]
        d : int
            fitting する多項式の次数
        """
        n = len(x)
        # ランダムサンプリングの件数は最小限（モデルの自由度と同じ）にする
        n_part = d+1
        mse_min = np.inf
        ls_best = None
        x_inliers_best = None
        y_inliers_best = None

        # Copy x and y to temporarry array (x_temp and y_temp is complete, x and y we will remove fix point)
        x_temp = x
        y_temp = y

        # Removing of fix point from x and y, but fix point is still in temp arrays
        x_temp = np.delete(x_temp, selectedIndex)
        y_temp = np.delete(y_temp, selectedIndex)

        for t in range(self.max_trials):
            index_list = np.random.choice(n-1, n_part-1, replace=False)
            x_part, y_part = x_temp[index_list], y_temp[index_list]
            
            # Add fix point to x_part and y_part
            x_part = np.append(x_part, [x[selectedIndex]])
            y_part = np.append(y_part, [y[selectedIndex]])

            ls_part = PolynomialLeastSquare()
            ls_part.fit(x_part, y_part, d)
            ids_inliers = self.__detect_inliers_indices(x, y, ls_part)
            if len(ids_inliers) / n < self.min_inliers_rate:
                continue
            x_inliers, y_inliers = x[ids_inliers], y[ids_inliers]
            ls_inliers = PolynomialLeastSquare()
            ls_inliers.fit(x_inliers, y_inliers, d)
            mse = self.__calc_mse(x_inliers, y_inliers, ls_inliers)
            if mse < mse_min:
                mse_min = mse
                ls_best = ls_inliers
                x_inliers_best, y_inliers_best = x_inliers, y_inliers
        self.mse = mse_min
        self.ls_best = ls_best
        self.x_inliers_best, self.y_inliers_best = x_inliers_best, y_inliers_best
    
    def __detect_inliers_indices(self, x, y, model):
        """
        回帰モデルに対する正常値（のインデックス）を見つける
        """
        y_pred = model.predict(x)
        return np.where(np.abs(y-y_pred) < self.residual_threshold)[0]
    
    def __calc_mse(self, x, y, model):
        """
        回帰モデルに対するデータの平均二乗誤差（MSE）を計算
        """
        y_pred = model.predict(x)
        return np.average((y_pred-y)**2)