import copy
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import panel as pn

from Ransac import RANSAC
from PolynomialLeastSquare import PolynomialLeastSquare

file_path = "./files/"


class RANSAC2D:
    df = []
    x = []
    y = []
    z = []
    selectedIndex = {}
    selectedPoint = {}
    rs_x_inliers_best = {}
    rs_y_inliers_best = {}

    def readCSV(self, file_path):
        self.df = pd.read_csv(file_path + "samplefile.csv")
    
    def assignValues(self, df):
        self.x = np.array(df['x'])
        self.y = np.array(df['y'])
        self.z = np.array(df['z'])

    def create_plot(self):
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=self.df['x'], y=self.df['y'], z=self.df['z'],
                    mode='markers',
                    marker=dict(size=1,
                                color = ['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(self.df['red'], self.df['green'], self.df['blue'])
                                ]),
                    ids=self.df['ID']
                )
            ],
        )
        fig.layout.autosize = True
        return fig

    def main_function(self):
        print({'x0' :self.x[selectedIndex], 'y0' :self.y[selectedIndex]})

        # RANSAC の fitting
        rs = RANSAC(max_trials=1000, residual_threshold=1.5, min_inliers_rate=0.4)
        rs.fit(self.x, self.y, 4, selectedIndex)
        # print({'平均二乗誤差' : rs.mse})
        # print({'RANSAC分類器' : rs.ls_best.coefficients})

        # 単純な最小二乗法の fitting
        ls = PolynomialLeastSquare()
        ls.fit(self.x, self.y, 4)
        # print({'LeastSquare分類器' : ls.coefficients})

        # 回帰曲線を描画
        xp = np.arange(self.df['x'].min(), self.df['x'].max(), 0.1)
        yp_rs = rs.predict(xp)  #Ransac
        yp_ls = ls.predict(xp)  #LeastSquare
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(12,4))
        axL.set_title('RANSAC with Least Square')
        axL.set_xlabel('$X$')
        axL.set_ylabel('$Y$')
        axL.scatter(self.x, self.y, s=10, label='outliers')
        axL.scatter(rs.x_inliers_best, rs.y_inliers_best, s=15, label='Inliers')
        axL.plot(xp, yp_rs, c='r', label='regression curve')
        axL.grid()
        axL.legend()
        axR.set_title('Simple Least Square')
        axR.set_xlabel('$X$')
        axR.set_ylabel('$Y$')
        axR.scatter(self.x, self.y, s=10)
        axR.plot(xp, yp_ls, c='r', label='regression curve')
        axR.grid()
        axR.legend()
        plt.show()

        return rs.x_inliers_best, rs.y_inliers_best
    
if __name__ == '__main__':
    ransac2d = RANSAC2D()
    pn.extension("plotly")
    pn.config.sizing_mode = "stretch_width"
    plot = ransac2d.create_plot()
    plot_panel = pn.pane.Plotly(plot, config={"responsive": True}, sizing_mode="stretch_both")
    @pn.depends(plot_panel.param.click_data, watch=True)
    def choose_point(click_data):
        global selectedPoint
        global selectedIndex
        global rs_x_inliers_best
        global rs_y_inliers_best
        selectedPoint = click_data
        selectedIndex = selectedPoint['points'][0]['pointNumber']
        rs_x_inliers_best, rs_y_inliers_best = ransac2d.main_function(selectedPoint, selectedIndex)

    app = pn.Column(plot_panel)

    app.servable()