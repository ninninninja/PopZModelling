"""
+----------------------------------------------------------------------------------+
|   Update time: 20170214-02:25 (Valentine's Day)                                  |
+----------------------------------------------------------------------------------+
"""

import numpy as np
from scipy.sparse import find
import random

class BehaveSet:
    def __init__(self):
        """
        :var self.DegCollector: store serial number of multimers which will have degraded.
        :var self.LastSerialNum: store the serial numbers which will be used in Binding Function.
        :var self.Guide: call a function to get the location of grids which are empty and able to be moved into.
        :var self.interact: call a function to get the location of grids which are occupied and able to bind with.
        :var self.Diffuse: call a function to execute diffusion.
        :var self.Bind: call a function to execute binding.
        """
        self.DegCollector = np.array([], dtype=int)
        self.LastSerialNum = int(1)
        self.Guide = self.GetNeighiborEmpty
        self.interact = self.GetNeighiborFilled
        self.Diffuse = self.Diffugo
        self.Bind = self.Bindgo

    def Scanning(self, QueCoordinate, PopZMap, threshold, NaNBoundary, NucleoReg):
        """
        :param QueCooordinate: the scanning priority of grids
        :param WhereISPopZ: where the grids are filled with PopZ
        :param threshold: it is an numpy array containing prod_prob, deg_prob, bind_prob and diff_prob
        :return:
        1) PopZMap: the final PopZ location after scanning whole grids
        """
        ###########################
        # Section 1:              #
        # Initilize all we need...#
        ###########################

        skip_array = []  # Record the grids which have been scanned...
        ProProb, DegProb, BindProb, DiffProb = threshold[0], threshold[1], threshold[2], threshold[3]
        ###########
        WhereISPopZ_arr = PopZMap
        ###########

        #############################
        # Section 2:                #
        # Start to scan all grids   #
        # which have been rearranged#
        #############################
        for coord in QueCoordinate:

            #############################
            # Section 2-1:              #
            # Check if the grids have   #
            # been scanned or not       #
            #############################

            # Find the location which has been scanned (string strategy)...
            str_coord = '{0},{1}'.format(coord[0], coord[1])
            if str_coord in skip_array:
                del skip_array[skip_array.index(str_coord)] # Delete elements if they have been found
                continue    # Skip this loop

            #############################
            # Section 2-2:              #
            # Execute behaviors...      #
            #############################

            # Get serial number if grids are filled
            SerialNumber = WhereISPopZ_arr[coord[0], coord[1]]

            ###############################
            # Empty & Generate a new PopZ #
            ###############################
            if SerialNumber == 0:
                if random.random() < ProProb:
                    WhereISPopZ_arr[coord[0], coord[1]] = 1  # Fill a new popz into empty grids

            #######################
            # -->Degradation      #
            #  -->Diffusion       #
            #       -->binding    #
            #######################
            # Monomer
            elif SerialNumber == 1:
                # Get location of monomers...
                location = np.array([coord[0], coord[1]])
                # Execute degradation, diffusion or nothing happen...
                if random.random() < DegProb:  # Monomer degrade
                    WhereISPopZ_arr[coord[0], coord[1]] = 0  # Local degrade
                elif random.random() < DiffProb:  # Monomer diffuse
                    # Find if there is any neighbor empty grid...
                    GuidePermit = self.Guide(WhereISPopZ_arr, NaNBoundary, NucleoReg, location)
                    #  GuidePermit is equal to zero indicates molecules staying.
                    if GuidePermit != 0:  # for example, 1010 or 0110...
                        PopZ_movement, locl_for_bind = self.Diffuse(WhereISPopZ_arr, GuidePermit,
                                                                    location)
                        skip_array.append('{0},{1}'.format(locl_for_bind[0], locl_for_bind[1])) # add a new location

                        # Update...
                        WhereISPopZ_arr = PopZ_movement

                        # Binding occurs when this molecule is not inside the nucleoid region...
                        if NucleoReg[locl_for_bind[0], locl_for_bind[1]] == 0:
                            # Find out the neighbor grids are filled and within a polymerization-allowed region.
                            # Bind_candidate: a integer which represents how many units can bind with
                            # locl_candidate: location of all candidates
                            HowManyBindingTar, locl_candidate = self.interact(BindProb, WhereISPopZ_arr, NucleoReg,
                                                                           locl_for_bind)
                            # I dug a hole and then making myself trapped in...
                            if HowManyBindingTar != 0:  # All of neighbor grids being empty is not true...
                                PopZ_assemble, locl_for_skip = self.Bindgo('All', WhereISPopZ_arr,
                                                                           locl_candidate, locl_for_bind)
                                tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in locl_for_skip]
                                skip_array = skip_array + tmplist
                                # Update...
                                WhereISPopZ_arr = PopZ_assemble

            # Multimer
            elif SerialNumber > 1:
                PolymerLocl = find(WhereISPopZ_arr == SerialNumber)
                location = np.array([coord[0], coord[1]])
                GuidePermit = self.Guide(WhereISPopZ_arr, NaNBoundary, NucleoReg, location, PolymerLocl)
                DiffProbEDIT = DiffProb / (len(PolymerLocl[0]) ** (1 / 2))
                if random.random() < DegProb / len(PolymerLocl[0]):
                    # if degradation occurs, whole polymer would be eliminated.
                    WhereISPopZ_arr[PolymerLocl[0], PolymerLocl[1]] = 0
                    tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in PolymerLocl]
                    skip_array = skip_array + tmplist
                    # Save the discarded serial numbers
                    self.DegCollector = np.append(self.DegCollector, SerialNumber)
                    self.DegCollector = np.unique(self.DegCollector)

                elif random.random() < DiffProbEDIT and GuidePermit != 0:  ###DiffuProb is depended on size!!!!
                    PopZ_movement, locl_for_bind = self.Diffuse(WhereISPopZ_arr, GuidePermit,
                                                                    location, PolymerLocl)
                    tmp_locl_for_bind = locl_for_bind.T
                    tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in tmp_locl_for_bind]
                    skip_array = skip_array + tmplist

                    # Update...
                    WhereISPopZ_arr = PopZ_movement

                    # Binding occurs when this molecule is not inside the nucleoid region...
                    if np.any(NucleoReg[locl_for_bind[0], locl_for_bind[1]]) == 0:
                        HowManyBindingTar, locl_candidate = self.interact(BindProb, WhereISPopZ_arr,
                                                                       NucleoReg, locl_for_bind)

                        if HowManyBindingTar != 0:
                            PopZ_assemble, locl_for_skip = self.Bindgo('All', WhereISPopZ_arr,
                                                                       locl_candidate, locl_for_bind)
                            tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in locl_for_skip]
                            skip_array = skip_array + tmplist
                            # Update...
                            WhereISPopZ_arr = PopZ_assemble
                else:
                    skip_array.append('{0},{1}'.format(coord[0], coord[1]))

        return WhereISPopZ_arr

    def GetNeighiborEmpty(self, WhereISPopZ_arr, NaNBoundary, NucleoReg, localization, PolymerLocl=None):
        """
        :param WhereISPopZ_csr: WhereISPopZ_csr
        :param NaNBoundary: NaNBoundary, indicate the space of cell
        :param NucleoReg: A region filled with chromosomal DNA is compacted
        :param localization: represent the location which is scanned now
        :param PolymerLocl: if the grids scanned now are parts of multimers, "Polymerlocl" is not None and the location
        of polymers.
        """
        NaN = NaNBoundary
        PopZ = WhereISPopZ_arr
        Nucleo = NucleoReg
        DiffuPermit = 0  # four-digit number

        row_Dir = np.array([-1, 1, 0, 0])
        col_Dir = np.array([0, 0, -1, 1])  # up down left right

        """
        If 'polymerlocl' is None, it indicates that there is a monomer in this grid.
        Following, we will sequentially check the neighbor grids.
        Once the grids are empty and not out of the boundary, molecules are allowed to move into.

        Otherwise, the information of localization of polymer is picked and saved into 'polymerlocl' with an array type.
        We can scan by the row or column to know that there is any space for polymer diffusion.
        Importantly, using the difference between every two elements in a row or column information array is to
        aviod the gaps or holes in polymers.

        Diffusion-allowed region:
        thousands  hundreds  tens   ones
            0         0       0      0
            up       down    left  right
        index 0    index 1 index 2 index 3
        """

        if PolymerLocl == None:
            i = 3
            for r, c in zip(row_Dir, col_Dir):
                try:
                    if PopZ[localization[0] + r, localization[1] + c] == 0 and NaN[
                                localization[0] + r, localization[1] + c] == 1:  # Bug here...
                        DiffuPermit += 10 ** i
                except:
                    DiffuPermit += 0
                finally:
                    i -= 1

        else:
            UP = 1
            DOWN = 1
            LEFT = 1
            RIGHT = 1

            # updown
            UniqueCol = np.unique(PolymerLocl[1])
            for col in UniqueCol:  # Scan one column in every loop
                ScanCol = np.where(PolymerLocl[1] == col)
                RowGrid = PolymerLocl[0][ScanCol]
                Continu = RowGrid[1:] - RowGrid[0:-1]

                if PopZ[min(RowGrid) - 1, col] != 0 or NaN[min(RowGrid) - 1, col] != 1 or Nucleo[
                            min(RowGrid) - 1, col] == 1:
                    UP = 0

                if PopZ[max(RowGrid) + 1, col] != 0 or NaN[max(RowGrid) + 1, col] != 1 or Nucleo[
                            max(RowGrid) + 1, col] == 1:
                    DOWN = 0

                if UP == 1 or DOWN == 1:
                    if np.all(Continu == 1) == False:
                        flaw, = np.where(Continu != 1)
                        col = np.repeat(col, len(flaw))  # Make "col" as an array having the same length as "flaw"
                        if np.any(PopZ[RowGrid[flaw + 1] - 1, col]) != 0 or np.all(
                                NaN[RowGrid[flaw + 1] - 1, col]) != 1 or np.any(Nucleo[
                                    RowGrid[flaw + 1] - 1, col]) == 1:
                            UP = 0

                        if np.any(PopZ[RowGrid[flaw] + 1, col]) != 0 or np.all(
                                NaN[RowGrid[flaw] + 1, col]) != 1 or np.any(Nucleo[
                                    RowGrid[flaw] + 1, col]) == 1:
                            DOWN = 0

            # left&right
            UniqueRow = np.unique(PolymerLocl[0])
            for row in UniqueRow:
                ScanRow = np.where(PolymerLocl[0] == row)
                ColGrid = PolymerLocl[1][ScanRow]
                Continu = ColGrid[1:] - ColGrid[0:-1]

                if PopZ[row, min(ColGrid) - 1] != 0 or NaN[row, min(ColGrid) - 1] != 1 or Nucleo[
                    row, min(ColGrid) - 1] == 1:
                    LEFT = 0


                if PopZ[row, max(ColGrid) + 1] != 0 or NaN[row, max(ColGrid) + 1] != 1 or Nucleo[
                    row, max(ColGrid) + 1] == 1:
                    RIGHT = 0

                if LEFT == 1 or RIGHT == 1:
                    if np.all(Continu == 1) is False:
                        flaw, = np.where(Continu != 1)
                        row = np.repeat(row, len(flaw))  # Make "row" as an array having the same length as "flaw"
                        if np.any(PopZ[row, ColGrid[flaw + 1] - 1]) != 0 or np.all(
                                NaN[row, ColGrid[flaw + 1] - 1]) != 1 or np.any(Nucleo[
                            row, ColGrid[flaw + 1] - 1]) == 1:
                            LEFT = 0

                        if np.any(PopZ[row, ColGrid[flaw] + 1]) != 0 or np.all(
                                NaN[row, ColGrid[flaw] + 1]) != 1 or np.any(Nucleo[
                            row, ColGrid[flaw] + 1]) == 1:
                            RIGHT = 0

            DiffuPermit = UP * 1000 + DOWN * 100 + LEFT * 10 + RIGHT * 1

        return DiffuPermit

    def Diffugo(self, WhereISPopZ_arr, DiffuPermit, localization, PolymerLocl=None):

        PopZ_arr = WhereISPopZ_arr

        row_Dir = np.array([-1, 1, 0, 0])
        col_Dir = np.array([0, 0, -1, 1])  # up down left right
        DiffuPermit = np.array([int(i) for i in str(DiffuPermit)])  # Take diffusion permits apart to an array
        if len(DiffuPermit) != 4:
            for d in range(4 - len(DiffuPermit)):
                DiffuPermit = np.append(0, DiffuPermit)

        DiffuChoose, = np.nonzero(DiffuPermit)  # 0: up, 1: down, 2: left, right: 3
        randnum = np.random.randint(len(DiffuChoose))
        Radd = row_Dir[DiffuChoose[randnum]]
        Cadd = col_Dir[DiffuChoose[randnum]]

        if PolymerLocl is None:  # monomer diffusion
            PopZ_arr[localization[0], localization[1]] = 0
            PopZ_arr[localization[0] + Radd, localization[1] + Cadd] = 1  # bug here...out of bounds
            locl_for_binding = np.array([localization[0] + Radd, localization[1] + Cadd])   # 1D array

        else:  # polymer diffusion
            SaveValue = PopZ_arr[PolymerLocl[0][0], PolymerLocl[1][0]]
            PopZ_arr[PolymerLocl[0], PolymerLocl[1]] = 0  # Elimiate the location of polymer
            PopZ_arr[PolymerLocl[0] + Radd, PolymerLocl[1] + Cadd] = SaveValue
            locl_for_binding = np.array([PolymerLocl[0] + Radd, PolymerLocl[1] + Cadd])     # 1D array

        return PopZ_arr, locl_for_binding

    def GetNeighiborFilled(self, BindingThreshold, WhereISPopZ_arr, NucleoReg, localization):
        """
        :param WhereISPopZ_csr: WhereISPopZ_csr
        :param NaNBoundary: NaNBoundary, indicate the space of cell
        :param NucleoReg:
        :param localization: New location after diffusion
        :param PolymerLocl:
        """
        PopZ = WhereISPopZ_arr
        Nucleo = NucleoReg
        Candidate_row = np.array([])  # Record the location of binding candidates
        Candidate_col = np.array([])
        row_Dir = np.array([-1, 1, 0, 0])
        col_Dir = np.array([0, 0, -1, 1])  # up down left right

        """
        If 'polymerlocl' is None, it indicates that there is a monomer in this grid.
        Following, we will sequentially check the neighbor grids.
        Once the grids are empty and not out of the boundary, molecules are allowed to move into.

        Otherwise, the information of localization of polymer is picked and saved into 'polymerlocl' with an array type.
        We can scan by the row or column to know that there is any space for polymer diffusion.
        Importantly, using the difference between every two elements in a row or column information array is to
        aviod the gaps or holes in polymers.
        """

        if localization.ndim == 1:
            for r, c in zip(row_Dir, col_Dir):
                try:
                    Num = PopZ[localization[0] + r, localization[1] + c]
                    Nuc = Nucleo[localization[0] + r, localization[1] + c]
                except:
                    continue

                # Avoid repeated recording the info of the same candidate
                if Num != 0 and Nuc != 1:  ### ValueError: The truth value of an array with more than one element is ambiguous.
                    # Record row info of binding candidates
                    Candidate_row = np.append(Candidate_row, localization[0] + r)
                    # Record column info of binding candidates
                    Candidate_col = np.append(Candidate_col, localization[1] + c)

        elif localization.ndim >= 2:
            # updown
            UniqueCol = np.unique(localization[1])
            for col in UniqueCol:  # Scan one column in every loop
                ScanCol = np.where(localization[1] == col)  # Find the grids in the same column in this multimer
                RowGrid = localization[0][ScanCol]  # Find the row indices corresponding to the column indices
                Continu = RowGrid[1:] - RowGrid[0:-1]  # To Check if there is any gap hiding in this column

                # UP
                try:
                    # Get serial number
                    NumUP = PopZ[min(RowGrid) - 1, col]
                except:
                    pass
                else:
                    if NumUP != 0 and Nucleo[min(RowGrid) - 1, col] != 1:
                        Candidate_row = np.append(Candidate_row, (min(RowGrid) - 1))
                        Candidate_col = np.append(Candidate_col, col)

                # DOWN
                try:
                    # Get serial number
                    NumDOWN = PopZ[max(RowGrid) + 1, col]
                except:
                    pass
                else:
                    if NumDOWN != 0 and Nucleo[max(RowGrid) + 1, col] != 1:
                        Candidate_row = np.append(Candidate_row, (max(RowGrid) + 1))
                        Candidate_col = np.append(Candidate_col, col)

                # gaps
                if np.all(Continu == 1) == False:
                    flaw, = np.where(Continu != 1)
                    for gap in flaw:  # There is only one gap between the two grids.
                        Num = PopZ[RowGrid[gap + 1] - 1, col]
                        if Num != 0 and Nucleo[RowGrid[gap + 1] - 1, col] != 1:
                            Candidate_row = np.append(Candidate_row, (RowGrid[gap + 1] - 1))
                            Candidate_col = np.append(Candidate_col, col)
                            if Continu[gap] > 2:
                                if PopZ[RowGrid[gap] + 1, col] != 0 and Nucleo[RowGrid[gap] + 1, col] == 0:
                                    Candidate_row = np.append(Candidate_row, (RowGrid[gap] + 1))
                                    Candidate_col = np.append(Candidate_col, col)

            # left&right
            UniqueRow = np.unique(localization[0])
            for row in UniqueRow:
                ScanRow = np.where(localization[0] == row)
                ColGrid = localization[1][ScanRow]
                Continu = ColGrid[1:] - ColGrid[0:-1]

                # Left
                try:
                    # Get serial number
                    NumLEFT = PopZ[row, min(ColGrid) - 1]
                except:
                    pass
                else:
                    if NumLEFT != 0 and Nucleo[row, min(ColGrid) - 1] != 1:
                        Candidate_row = np.append(Candidate_row, row)
                        Candidate_col = np.append(Candidate_col, min(ColGrid) - 1)

                # Right
                try:
                    # Get serial number
                    NumRIGHT = PopZ[row, max(ColGrid) + 1]
                except:
                    pass
                else:
                    if NumRIGHT != 0 and Nucleo[row, max(ColGrid) + 1] != 1:
                        Candidate_row = np.append(Candidate_row, row)
                        Candidate_col = np.append(Candidate_col, max(ColGrid) + 1)

                # gaps
                if np.all(Continu == 1) is False:
                    flaw, = np.where(Continu != 1)
                    for gap in flaw:
                        Num = PopZ[row, ColGrid[gap + 1] - 1]
                        if Num != 0 and Nucleo[row, ColGrid[gap + 1] - 1] == 0:
                            Candidate_row = np.append(Candidate_row, row)
                            Candidate_col = np.append(Candidate_col, ColGrid[gap + 1] - 1)
                            if Continu[gap] > 2:
                                if PopZ[row, ColGrid[gap] + 1] != 0 and Nucleo[row, ColGrid[gap] + 1] == 0:
                                    Candidate_row = np.append(Candidate_row, row)
                                    Candidate_col = np.append(Candidate_col, ColGrid[gap] + 1)

        Candidate_locl = np.array([Candidate_row, Candidate_col], dtype=int)   # array([[row], [col]])

        if len(Candidate_row)>0:
            WaitforDelete = []
            for index in range(len(Candidate_row)):
                if random.random() > BindingThreshold:
                    WaitforDelete.append(index)
            Candidate_locl = np.delete(Candidate_locl, WaitforDelete, axis=1)

        HowManyBindingTar = Candidate_locl.shape[1]

        return HowManyBindingTar, Candidate_locl


    def Bindgo(self, Binding_Method, WhereISPopZ_arr, Candidate_locl, localization):

        locl_target = []  # To solve UnboundLocalError

        PopZ_arr = WhereISPopZ_arr

        # All serial number of binding target
        SerialNumofTar = np.array(PopZ_arr[Candidate_locl[0], Candidate_locl[1]])

        if Binding_Method == 'All':
            # self is monomer
            if localization.squeeze().ndim == 1:
                # monomer binding target
                if np.all(SerialNumofTar==1):
                    locl_target = np.column_stack((Candidate_locl[0], Candidate_locl[1]))
                    # Give this new multimer with a serial number which was discarded.
                    if len(self.DegCollector) > 0:
                        WatingNum = min(self.DegCollector)
                        PopZ_arr[Candidate_locl[0], Candidate_locl[1]] = WatingNum
                        PopZ_arr[localization[0], localization[1]] = WatingNum
                        self.DegCollector = np.delete(self.DegCollector, 0)
                    # Give this new multimer with a new serial number.
                    else:
                        PopZ_arr[Candidate_locl[0], Candidate_locl[1]] = self.LastSerialNum + 1
                        PopZ_arr[localization[0], localization[1]] = self.LastSerialNum + 1
                        self.LastSerialNum += 1

                # Polymer binding target
                else:
                    fordel, = np.where(SerialNumofTar == 1)
                    if len(fordel) != 0:    # There is at least one monomer as binding target.
                        list_row = Candidate_locl[0][fordel].astype(int)
                        list_col = Candidate_locl[1][fordel].astype(int)
                        SerialNumofTar = np.unique(np.delete(SerialNumofTar, fordel)) # Delete the serial number 1
                    else:
                        SerialNumofTar = np.unique(SerialNumofTar)
                        list_row, list_col = np.array([], dtype=int), np.array([], dtype=int)

                    if len(SerialNumofTar) > 1:
                        # Appending binding target's coordinates
                        for i in range(1, len(SerialNumofTar)):
                            # Put the discarded SN to DegCollector
                            self.DegCollector = np.append(self.DegCollector, SerialNumofTar[i])
                            # In every time, the first number of SerialNumofTar is used to be serial number.
                            tmp_row, tmp_col = np.where(PopZ_arr == SerialNumofTar[i])
                            list_row = np.append(list_row, tmp_row)
                            list_col = np.append(list_col, tmp_col)
                        # Avoid get repeated serial number from DegCollector
                        self.DegCollector = np.unique(self.DegCollector)
                        # All serial numbers of binding targets are going to be the same as the first SN.
                        PopZ_arr[list_row, list_col] = SerialNumofTar[0]
                    # ...including local grid
                    PopZ_arr[localization[0], localization[1]] = SerialNumofTar[0]
                    # These coordinates are recorded and then skipped while scanning grids.
                    locl_target = np.column_stack((list_row, list_col))

            # self is polymer
            else:
                # monomer binding target
                if np.all(SerialNumofTar==1):
                    PopZ_arr[Candidate_locl[0], Candidate_locl[1]] = PopZ_arr[
                        localization[0][0], localization[1][0]]
                    locl_target = np.column_stack((Candidate_locl[0], Candidate_locl[1]))

                # Polymer binding target
                else:
                    fordel, = np.where(SerialNumofTar == 1)
                    if len(fordel) != 0:
                        list_row = Candidate_locl[0][fordel].astype(int)
                        list_col = Candidate_locl[1][fordel].astype(int)
                        SerialNumofTar = np.unique(np.delete(SerialNumofTar, fordel))  # Delete the serial number 1
                    else:
                        SerialNumofTar = np.unique(SerialNumofTar)
                        list_row, list_col = np.array([], dtype=int), np.array([], dtype=int)

                    # Appending binding target's coordinates
                    for SN in SerialNumofTar:
                        self.DegCollector = np.append(self.DegCollector, SN)   # Put the discarded SN to DegCollector
                        tmp_row, tmp_col = np.where(PopZ_arr == SN)
                        list_row = np.append(list_row, tmp_row)
                        list_col = np.append(list_col, tmp_col)
                    # Avoid get repeated serial number from DegCollector
                    self.DegCollector = np.unique(self.DegCollector)
                    # Reshape the skip array to two dimensional array
                    locl_target = np.column_stack((list_row, list_col))
                    # Change the serial number of binding targets to local SN
                    PopZ_arr[list_row, list_col] = PopZ_arr[localization[0][0], localization[1][0]]


        elif Binding_Method == 'Old':
            randnum = random.randint(0, len(Candidate_locl[0])-1)
            # self is monomer
            if localization.squeeze().ndim == 1:
                # monomer binding target
                if PopZ_arr[Candidate_locl[0][randnum], Candidate_locl[1][randnum]] == 1:
                    locl_target = np.array([[Candidate_locl[0][randnum], Candidate_locl[1][randnum]]])
                    if len(self.DegCollector) > 0:
                        WatingNum = min(self.DegCollector)
                        PopZ_arr[Candidate_locl[0][randnum], Candidate_locl[1][randnum]] = WatingNum
                        PopZ_arr[localization[0], localization[1]] = WatingNum
                        self.DegCollector = np.delete(self.DegCollector, 0)

                    else:
                        PopZ_arr[Candidate_locl[0][randnum], Candidate_locl[1][randnum]] = self.LastSerialNum + 1
                        PopZ_arr[localization[0], localization[1]] = self.LastSerialNum + 1
                        self.LastSerialNum += 1

                # Polymer binding target
                else:
                    locl_target = np.where(PopZ_arr == PopZ_arr[Candidate_locl[0][randnum], Candidate_locl[1][randnum]])
                    locl_target = np.column_stack((locl_target[0], locl_target[1]))
                    PopZ_arr[localization[0], localization[1]] = PopZ_arr[
                        Candidate_locl[0][randnum], Candidate_locl[1][randnum]]

            # self is polymer
            else:
                # monomer binding target
                if PopZ_arr[Candidate_locl[0][randnum], Candidate_locl[1][randnum]] == 1:
                    PopZ_arr[Candidate_locl[0][randnum], Candidate_locl[1][randnum]] = PopZ_arr[
                        localization[0][0], localization[1][0]]
                    locl_target = np.array([[Candidate_locl[0][randnum], Candidate_locl[1][randnum]]])

                # Polymer binding target
                else:
                    target = find(PopZ_arr == PopZ_arr[Candidate_locl[0][randnum], Candidate_locl[1][randnum]])
                    self.DegCollector = np.append(self.DegCollector, PopZ_arr[target[0][0], target[1][0]])
                    self.DegCollector = np.unique(self.DegCollector)
                    PopZ_arr[target[0], target[1]] = PopZ_arr[localization[0][0], localization[1][0]]
                    locl_target = np.column_stack((target[0], target[1]))

        return PopZ_arr, locl_target  ### UnboundLocalError: local variable 'locl_target' referenced before assignment

    def Monitor(self, NucleoReg, PopZMap, coord):
        permission = False
        tmpMap = PopZMap.copy()
        mark = np.where(NucleoReg == 0)[0]
        tmpMap[mark[0], mark[1]] = 0
        PopZinNucleo = np.sum(tmpMap)
        if PopZinNucleo < 0.15*np.sum(NucleoReg):
            SN = PopZMap[coord[0], coord[1]]
            getSize = len(np.where(PopZMap==SN)[0])
            if 0.15*np.sum(NucleoReg) - PopZinNucleo - getSize >=0:
                permission = True

        return PopZinNucleo
