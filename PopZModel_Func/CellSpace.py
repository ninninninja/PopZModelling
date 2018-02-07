import numpy as np
from collections import OrderedDict
from scipy.sparse import coo_matrix

"""
All scale of grids are compared to PopZ.
update time: 20170618-03:42
"""

class CellSpaceSet:
    def __init__(self, CellSize, accomd):
        # ori cell size
        self.xlim = CellSize[0]
        self.ylim = CellSize[1]  # the same as diameter
        self.accomd = accomd
        self.amp = 0    # There is no amplified scale of grid in the beginning.
        self.emptygrid = {}  # init the cell

    def amplify(self):  # amplification
        # the accommodation of every grid
        totalgrid = (1.5*10 ** 5) // self.accomd

        # amplify the grid size
        self.amp = int(totalgrid // 7.14)  # pi*(ylim/2)^2+ylim*xlim

        return self.amp

    def findNaN(self):
        cellspace = self.emptygrid['cell composition']
        row = np.array([0, 0, 1, -1])
        col = np.array([1, -1, 0, 0])

        nanUD = np.zeros((1, cellspace.shape[1]))*np.nan
        nanLR = np.zeros((cellspace.shape[0]+2, 1))*np.nan

        cellspace = np.concatenate((nanUD, cellspace, nanUD), axis=0)
        cellspace = np.concatenate((nanLR, cellspace, nanLR), axis=1)

        mem_reg = cellspace.copy()*np.nan
        for i in range(cellspace.shape[0]):
            for j in range(cellspace.shape[1]):
                if np.isnan(cellspace[i, j]) == 0:
                    for r, c in zip(row, col):
                        try:
                            if np.isnan(cellspace[i+r, j+c]) == 1 and i+r >= 0 and j+c >= 0:
                                mem_reg[i+r, j+c] = 0
                        except:
                            continue

        self.emptygrid['cell composition'] = cellspace
        self.emptygrid['Membrane'] = mem_reg

    def gridsetting(self):
        # Get information needed for creating cell space
        amp = self.amp
        xlim = self.xlim
        ylim = self.ylim
        nucleo_radius = int(round((0.7*amp**2)**(1/2)))

        # helper matrix
        circgrid_ori = np.zeros((amp, amp))     # edit for reshaping 6/14

        # init the grid in semicircle region
        circreg_LU = circgrid_ori.copy()  # left up

        # init the square region with grid
        sqrreg_Up = np.zeros((amp * (xlim - ylim) // 2, (2* amp * ylim // 2))) # Up # edit for reshaping 6/14
        sqrreg_Up[(sqrreg_Up.shape[0]-nucleo_radius):, :] = 2 # nucleoid region
        sqrreg_Dn = np.flipud(sqrreg_Up) # Down
        sqrfinal = np.concatenate((sqrreg_Up, sqrreg_Dn), axis=0) # Assemble up and down square matrix

        # Define the semicircular regions.
        for i in range(0, amp):
            y = i
            # calulate how many grids in semicircle regions
            column_len = int(round(((amp ** 2) - (y ** 2)) ** (1 / 2)))
            #column_len_semi = int(round((2.25*((amp ** 2) - (y ** 2))) ** (1 / 2)))# edit for reshaping 6/14
            if (nucleo_radius ** 2) >= (y**2):
                nucleo_col = int(round(((nucleo_radius ** 2) - (y ** 2)) ** (1 / 2)))
            else:
                nucleo_col = 0

            # define the grid in semicircle regions
            circreg_LU[amp - 1 - i, 0:(amp - column_len)] = np.nan# edit for reshaping 6/14
            circreg_LU[amp - 1 - i, (amp - nucleo_col):] = 2# edit for reshaping 6/14

        circreg_LD = np.flipud(circreg_LU)  # left down
        circreg_RU = np.fliplr(circreg_LU)  # right up
        circreg_RD = np.flipud(circreg_RU)  # right down

        """
        Simple calculating system in only one region
        """
        # Merge the left and right semicircle
        #  n n n n n
        #  n n n n 0
        #  n n n 0 0
        #  n n n 0 0
        #  n n 0 0 2
        #  n 0 0 2 2
        #  0 0 2 2 2
        #  0 0 2 2 2
        #  n 0 0 2 2
        #  n n 0 0 2
        #  n n n 0 0
        #  n n n n 0
        #  n n n n n
        row_len = 2*circgrid_ori.shape[0]
        NaNarray_LR = np.nan * np.ones((row_len, 1))
        semimerge_left = np.concatenate((circreg_LU, circreg_LD), axis=0)
        semimerge_left = np.concatenate((NaNarray_LR, semimerge_left), axis=1)
        semimerge_right = np.concatenate((circreg_RU, circreg_RD), axis=0)
        semimerge_right = np.concatenate((semimerge_right, NaNarray_LR), axis=1)
        celltmpspace = np.concatenate((semimerge_left, sqrfinal, semimerge_right), axis=1)
        NaNarray_UD = np.nan * np.ones((1, celltmpspace.shape[1]))
        celltmpspace = np.concatenate((NaNarray_UD, celltmpspace, NaNarray_UD), axis=0)

        # Set density of the cell
        density = 0.75

        # Save all information with a ordered dictionary
        cellallspace = OrderedDict()
        cellallspace['cell composition'] = celltmpspace
        cellallspace['density'] = density
        #cellallspace['Membrane'] = MemSpace

        # Change the property of emptygrid
        self.emptygrid = cellallspace

        return cellallspace

    def Sparse_Cell(self):
        WholeCell = self.emptygrid['cell composition']

        # Sparse Matrix
        # Creat a empty cell space
        cytoDim = WholeCell.shape
        EmptyCell = coo_matrix((cytoDim[0], cytoDim[1]), dtype=np.int8)

        # Determine where is nucleoid region
        coord_2nd = np.where(WholeCell==2)
        coord_r = coord_2nd[0]
        coord_c = coord_2nd[1]
        mid = np.median(np.unique(coord_c))
        del_range = 10
        delpos = [mid + 0.5 + plus for plus in range(del_range)]
        delneg = [mid - 0.5 - minus for minus in range(del_range)]

        for pos in delpos:
            getridof, = np.where(coord_c==pos)
            coord_r = np.delete(coord_r, getridof)
            coord_c = np.delete(coord_c, getridof)

        for neg in delneg:
            getridof, = np.where(coord_c==neg)
            coord_r = np.delete(coord_r, getridof)
            coord_c = np.delete(coord_c, getridof)

        for wh, coltmp in enumerate(coord_c):
            if coltmp < mid:
                coord_c[wh] += del_range
            elif coltmp > mid:
                coord_c[wh] -= del_range


        value = np.ones((len(coord_r)))
        NucleoReg = coo_matrix((value, (coord_r, coord_c)),
                               shape=(cytoDim[0], cytoDim[1]), dtype=bool)

        # Define the boundary of cell by NaN
        coord_3rd = np.where(np.isnan(WholeCell)==0)
        v = np.ones((len(coord_3rd[0])))
        NaNBoundary = coo_matrix((v, (coord_3rd[0], coord_3rd[1])),
                                 shape=(cytoDim[0], cytoDim[1]), dtype=bool)

        # Convert cell membrane
        Membrane = self.emptygrid['Membrane']
        Mem_coo = coo_matrix(Membrane)

        # Put all information into a matrix
        infoCell = np.ones((NaNBoundary.shape[0], NaNBoundary.shape[1]))
        infoCell[Mem_coo.row, Mem_coo.col] = 0
        infoCell[NaNBoundary.row, NaNBoundary.col] = 1
        infoCell_coo = coo_matrix(infoCell)

        # Save all information with a ordered dictionary
        # Three sparse matrix
        cooCellSpace = OrderedDict()
        cooCellSpace['EmptyCell'] = EmptyCell
        cooCellSpace['NucleoReg'] = NucleoReg
        cooCellSpace['NaNBoundary'] = NaNBoundary
        cooCellSpace['Membrane'] = Mem_coo
        cooCellSpace['DoableReg'] = infoCell_coo

        return cooCellSpace

    def Scheduling(self, NaNBoundary):
        EmptySparCell = NaNBoundary
        row, col = EmptySparCell.row, EmptySparCell.col

        lit = 0
        queueRow = np.zeros(len(row), dtype=int)
        queueCol = np.zeros(len(col), dtype=int)
        while len(row)>0:
            index = np.random.randint(len(row))
            queueRow[lit], queueCol[lit] = row[index], col[index]
            row, col = np.delete(row, index), np.delete(col, index)
            lit +=1

        return queueRow, queueCol
