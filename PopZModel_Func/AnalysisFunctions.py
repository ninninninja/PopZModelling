"""
Update: 20170325
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, find
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import _pickle as cPickle
from sklearn.utils.extmath import cartesian

class Realtime_Display:
    def __init__(self):
        self.pic_image = None
        self.x_line = None
        self.y_line = None
        self.pic_text = None
        self.xdata = None
        self.ydata = None

    def init_figure(self, PopZMap, NaN, init_t):
        # Construct a normal 2D array for plotting
        Round = find(NaN == 0)
        DrawMap = np.zeros((PopZMap.shape[0], PopZMap.shape[1]), dtype=int)
        DrawMap[Round[0], Round[1]] = 10  # Boundary, maximum value in the cell

        # Two variable are used to record distribution pattern
        ydataori = np.zeros(PopZMap.shape[0])
        self.ydata = ydataori.copy()
        xdataori = np.zeros(PopZMap.shape[1])
        self.xdata = xdataori.copy()

        y_r, y_d = np.unique(PopZMap.row, return_counts=True)
        x_c, x_d = np.unique(PopZMap.col, return_counts=True)
        self.ydata[y_r] = ydataori[y_r] + y_d
        self.xdata[x_c] = xdataori[x_c] + x_d

        x_xaxis = np.arange(len(self.xdata))
        y_xaxis = np.arange(len(self.ydata))

        # Open a new figure frame
        fig = plt.figure()

        # Subplot the polarity pattern along with x-axis
        ax_x = fig.add_axes([0.1, 0.02, 0.4, 0.3])
        ax_x.axis([0, PopZMap.shape[1], 0, PopZMap.shape[0]])
        ax_x.set_title('X axis pattern')

        # Subplot the polarity pattern along with y-axis
        ax_y = fig.add_axes([0.55, 0.02, 0.4, 0.3])
        ax_y.axis([0, PopZMap.shape[0], 0, PopZMap.shape[1]])
        ax_y.set_title('Y axis pattern')

        # Subplot PopZ distribution in cell space in real-time
        ax = fig.add_axes([0.15, 0.4, 0.7, 0.5])
        ax.set_title('Whole cell space')

        # Draw something on the figure
        x_line, = ax_x.plot(x_xaxis, self.xdata)
        y_line, = ax_y.plot(y_xaxis, self.ydata)
        pic_image = ax.imshow(DrawMap, cmap=cm.hot, vmax=10, vmin=0)  # cmap = cm.gray

        # Essential information display
        data_tmp = PopZMap.data.copy()
        ones_ind = np.where(data_tmp == 1)
        data_tmp[ones_ind] = 0
        info = ('Time: {0} \n'
                'Number of Polymers: {1} \n'.format(init_t, len(np.unique(data_tmp))))
        pic_text = ax.text(3, -2, info)

        self.pic_image = pic_image
        self.x_line = x_line
        self.y_line = y_line
        self.pic_text = pic_text

        return DrawMap, xdataori, ydataori

    def PopZDistribution(self, DrawMap, NewMap, xdataori, ydataori, time):
        # To avoid replacing original empty matrix, copying DrawMap as DrawtmpMap is a convenient way.
        DrawtmpMap = DrawMap.copy()

        # Update & show the results
        DrawtmpMap[NewMap.row, NewMap.col] = 1
        self.pic_image.set_data(DrawtmpMap)

        # Essential information display
        data_tmp = NewMap.data.copy()
        ones_ind = np.where(data_tmp == 1)
        data_tmp[ones_ind] = 0
        info = ('Time: {0} \n'
                'Number of PopZ: {1} \n'.format(time, len(np.unique(data_tmp))))
        self.pic_text.set_text(info)

        # Update & display the PopZ distributions which are along with x or y axis.
        y_r, y_d = np.unique(NewMap.row, return_counts=True)
        x_c, x_d = np.unique(NewMap.col, return_counts=True)
        self.ydata[y_r] = ydataori[y_r] + y_d
        self.xdata[x_c] = xdataori[x_c] + x_d
        self.x_line.set_ydata(self.xdata)
        self.y_line.set_ydata(self.ydata)

        # plt.pause(0.2)


class Statistcal_Analysis:
    def __init__(self):
        pass

    # Sampling
    # Latin Hypercube sampling
    def LHS(self, UpperLimit, LowerLimit, SampleNum, ParaMode=None):
        try:
            if len(UpperLimit) != len(LowerLimit):
                raise ValueError
        except ValueError as e:
            print(e.args)
            Print('The UpperLimit and LowerLimit must have same size of elements.')
            exit()

        if len(UpperLimit) == 1:
            para = np.zeros(SampleNum)

            if ParaMode == 'log scale':
                tmpfunc = np.logspace
            else:
                tmpfunc = np.linspace

            ParaSet = tmpfunc(UpperLimit, LowerLimit, SampleNum ** 2)
            ParaSet = ParaSet.reshape(SampleNum, SampleNum)

            # Start picking sample points in every region
            for ind in range(SampleNum):
                para[ind] = np.random.choice(ParaSet)

            # Shuffle sample points
            para = np.random.shuffle(para)

        else:
            para = np.zeros((SampleNum, len(UpperLimit)))
            for mode, itr in zip(ParaMode, range(len(UpperLimit))):
                if mode == 'log scale':
                    tmpfunc = np.logspace
                else:
                    tmpfunc = np.linspace

                ParaSet = tmpfunc(UpperLimit[itr], LowerLimit[itr], SampleNum ** 2)
                ParaSet = ParaSet.reshape(SampleNum, SampleNum)

                # Start picking sample points in every region
                for ind in range(SampleNum):
                    para[ind, itr] = np.random.choice(ParaSet[ind, :])

                # Shuffle sample points
                np.random.shuffle(para[:, itr])
        with open('LHS_Parameter.pickle', 'wb') as picklefile:
            cPickle.dump(para, picklefile, True)
            #==========LHS_Parameter.pickle==============#
            #[[Prod, Deg, Bind, Diffu
            #........................
            #........................
            # .......................]]
        print(para)
        return para

    # Ordinary Sampling, which is suitable for linear screening one parameter or fix at a point...
    def ODS(self, UpperLimit, LowerLimit, SampleNum, ParaMode=None):
        try:
            if len(UpperLimit) != len(LowerLimit):
                raise ValueError
        except ValueError as e:
            print(e.args)
            Print('The UpperLimit and LowerLimit must have same size of elements.')
            exit()

        if len(UpperLimit) == 1:
            para = np.zeros((SampleNum, 1))
            if ParaMode == 'log scale':
                tmpfunc = np.logspace
            else:
                tmpfunc = np.linspace

            ParaSet = tmpfunc(UpperLimit, LowerLimit, SampleNum)
            para[:, 0] = ParaSet

        else:
            para = np.zeros((SampleNum, len(UpperLimit)))
            for mode, itr in zip(ParaMode, range(len(UpperLimit))):
                if mode == 'log scale':
                    tmpfunc = np.logspace
                else:
                    tmpfunc = np.linspace

                ParaSet = tmpfunc(UpperLimit[itr], LowerLimit[itr], SampleNum)
                para[:, itr] = ParaSet


        with open('ODS_Parameter.pickle', 'wb') as picklefile:
            cPickle.dump(para, picklefile, True)
            #==========LHS_Parameter.pickle==============#
            #[[Prod, Deg, Bind, Diffu
            #........................
            #........................
            # .......................]]
        print(para)
        return para

    # Cross Sampling
    def CRS(self, UpperLimit, LowerLimit, SampleNum, ParaMode=None):
        try:
            if len(SampleNum) == 1:
                raise ValueError
        except ValueError as e:
            print(e.args)
            print('The parameter sets have to be at least two dimension.')
            exit()

        paratmp = np.zeros(len(UpperLimit), dtype=object)
        for itr, mode in enumerate(ParaMode):
            if mode == 'log scale':
                tmpfunc = np.logspace
            else:
                tmpfunc = np.linspace

            ParaSet = tmpfunc(UpperLimit[itr], LowerLimit[itr], SampleNum[itr])
            paratmp[itr] = ParaSet

        para = cartesian([paratmp[0], paratmp[1]])

        with open('CRS_Parameter.pickle', 'wb') as picklefile:
            cPickle.dump(para, picklefile, True)
            #==========LHS_Parameter.pickle==============#
            #[[Prod, Deg, Bind, Diffu
            #........................
            #........................
            # .......................]]
        print(para)
        return para




if __name__ == '__main__':
    s = Statistcal_Analysis()
    res = s.recepfunc(3)
    print(res)