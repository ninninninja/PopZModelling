"""
+-----------------------------------------------------------------------------------+
|   Update time: 20170618-12:12                                                     |
|   Version: 4.0                                                                    |
|   Update content:                                                                 |
|       1) Membrane protein, SpmX, which is able to bind with PopZ is added into
|          the model setting.
            --> The amount of SpmX is consistent without production and degradation.
            --> A 1 layer space where SpmX can move as in a 1D space is membrane.
            --> SpmX is capable of binding with multiple PopZ particles.
            --> While the SpmX and PopZ are in the same complex

    P.S. WTF... I spend almost one month to debug...
+-----------------------------------------------------------------------------------+
"""
import numpy as np
from scipy.sparse import find
import random
from sklearn.utils.extmath import cartesian


class BehaveSet:
    def __init__(self, BindGrowthFunc, DegDecayFunc):
        """
        :var self.DegCollector: store serial number of multimers which will have degraded.
        :var self.LastSerialNum: store the serial numbers which will be used in Binding Function.
        :var self.Guide: call a function to get the location of grids which are empty and able to be moved into.
        :var self.interact: call a function to get the location of grids which are occupied and able to bind with.
        :var self.Diffuse: call a function to execute diffusion.
        :var self.Bind: call a function to execute binding.
        """
        self.DegCollector = np.uint8(np.array([]))
        self.LastSerialNum = int(1)
        self.Guide = self.GetNeighiborEmpty
        self.interact = self.GetNeighiborFilled
        self.Diffuse = self.Diffugo
        self.Bind = self.Bindgo
        self.Monitor = self.Monitor
        self.edge = self.GetEdge
        self.Deg = self.Degradego
        self.ReAssignSN = self.ReAssignSN
        self.DegDecay = DegDecayFunc
        self.BindGrowth = BindGrowthFunc
        self.SingleSpmXGuide = self.SingleSpmXGuide

    def generateSpmX(self, SpmXMap, Amount):
        memreg = np.nonzero(np.isnan(SpmXMap) == 0)
        row, col = memreg[0], memreg[1]
        for _ in range(int(Amount)):
            index = np.random.randint(len(row))
            SpmXMap[row[index], col[index]] = 1
            row, col = np.delete(row, index), np.delete(col, index)

        return SpmXMap

    def Scanning(self, QueCoordinate, SpmXMap, PopZMap, threshold, NaNBoundary, NucleoReg):
        """
        :param QueCooordinate: the scanning priority of grids
        :param WhereISPopZ: where the grids are filled with PopZ
        :param threshold: it is an numpy array containing prod_prob, deg_prob, bind_prob and diff_prob
        :return:
        1) PopZMap: the final PopZ location after scanning whole grids
        """
        ############################
        # Section 1:               #
        # Initialize all we need...#
        ############################

        skip_array = []  # Record the grids which have been scanned...
        ProProb, DegProb, BindProb, DiffProb = threshold[0], threshold[1], threshold[2], threshold[3]
        ###########
        WhereISPopZ_arr = np.uint8(PopZMap)
        WhereISSpmX_arr = np.float16(SpmXMap)
        ###########
        Cooperativity = 'On'

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
                del skip_array[skip_array.index(str_coord)]  # Delete elements if they have been found
                continue  # Skip this loop

            #############################
            # Section 2-2:              #
            # Execute behaviors...      #
            #############################

            ###############################
            # Empty & Generate a new PopZ #
            ###############################

            # Get serial number if grids are filled
            SpmXgo = 0
            PopZgo = 0
            check_boundary = NaNBoundary[coord[0], coord[1]]
            if check_boundary == 1:  # Once encountering Nan values, it means the grid belongs to membrane.
                SerialNumber = WhereISPopZ_arr[coord[0], coord[1]]
                PopZgo = 1
            else:
                SerialNumber = WhereISSpmX_arr[coord[0], coord[1]]
                SpmXgo = 1

            ###############################
            # Empty & Generate a new PopZ #
            ###############################

            if PopZgo == 1 and SerialNumber == 0:
                if random.random() < ProProb:
                    if NucleoReg[coord[0], coord[1]] == 1:
                        if self.Monitor(NucleoReg, WhereISPopZ_arr, coord, 'Production') == True:
                            WhereISPopZ_arr[coord[0], coord[1]] = 1  # Fill a new popz into empty grids
                    else:
                        WhereISPopZ_arr[coord[0], coord[1]] = 1  # Fill a new popz into empty grids

            #######################
            # -->Degradation      #
            #  -->Diffusion       #
            #       -->binding    #
            #######################
            # Monomer
            elif SerialNumber == 1 and PopZgo == 1:
                # Get location of monomers...
                location = np.array([coord[0], coord[1]])
                # Execute degradation, diffusion or nothing happen...
                # Monomers have four units of surface which are capable of contacting with protease.
                DiffuAllow = 1
                for _ in range(4):
                    if random.random() < 10 * DegProb:  # Monomer degrade
                        WhereISPopZ_arr[coord[0], coord[1]] = 0  # Local degrade
                        DiffuAllow = 0
                        break
                if DiffuAllow == 1 and random.random() < DiffProb:  # Monomer diffuse
                    # Find if there is any neighbor empty grid...
                    GuidePermit = self.Guide('PopZ', WhereISPopZ_arr, NaNBoundary, NucleoReg, location)
                    #  GuidePermit is equal to zero indicates molecules staying.
                    if GuidePermit != 0:  # for example, 1010 or 0110...
                        PopZ_movement, locl_for_bind = self.Diffuse(WhereISPopZ_arr, GuidePermit,
                                                                    location)
                        skip_array.append('{0},{1}'.format(locl_for_bind[0], locl_for_bind[1]))  # add a new location

                        # Update...
                        WhereISPopZ_arr = PopZ_movement

                        # Binding occurs when this molecule is not inside the nucleoid region...
                        if NucleoReg[locl_for_bind[0], locl_for_bind[1]] == 0:
                            ParticleMap = {'PopZ': WhereISPopZ_arr, 'SpmX': WhereISSpmX_arr}
                            # Find out the neighbor grids are filled and within a polymerization-allowed region.
                            # Bind_candidate: a integer which represents how many units can bind with
                            # locl_candidate: location of all candidates
                            HowManyBindingTar, locl_candidate, SpmX_for_bind = self.interact('PopZ', BindProb,
                                                                                             ParticleMap, NucleoReg,
                                                                                             locl_for_bind,
                                                                                             Cooperativity)
                            # I dug a hole and then making myself trapped in...
                            if HowManyBindingTar != 0:  # All of neighbor grids being empty is not true...
                                PopZ_assemble, locl_for_skip = self.Bindgo('monoPopZ', WhereISPopZ_arr,
                                                                           locl_candidate, locl_for_bind,
                                                                           WhereISSpmX_arr, SpmX_for_bind)
                                tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in locl_for_skip]
                                skip_array = skip_array + tmplist
                                # Update...
                                WhereISPopZ_arr = PopZ_assemble[0]
                                WhereISSpmX_arr = PopZ_assemble[1]

            # Multimer
            elif SerialNumber > 1 and PopZgo == 1:
                PolymerLocl = np.nonzero(WhereISPopZ_arr == SerialNumber)
                SpmXLocl = np.nonzero(WhereISSpmX_arr == SerialNumber)
                SelfSize = len(PolymerLocl[0])
                location = np.array([coord[0], coord[1]])
                # Record coordinates for skipping
                Poly_skip_arr = np.column_stack((PolymerLocl[0], PolymerLocl[1]))
                tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in Poly_skip_arr]
                skip_array = skip_array + tmplist
                # Degradation...
                DegCandidate = self.edge(WhereISPopZ_arr, DegProb, PolymerLocl)
                if len(DegCandidate[0]) != 0:  # !=0
                    # Update...
                    DegResult = self.Degradego(WhereISPopZ_arr, DegCandidate, SerialNumber, WhereISSpmX_arr)
                    WhereISPopZ_arr = DegResult[0]
                    WhereISSpmX_arr = DegResult[1]

                # Find empty grids...
                else:  # len(DegCandidate[0]) == 0 indicates no target to be degraded
                    # Precomputation of diffusion
                    if len(SpmXLocl[0]) > 0:
                        GuidePermit = self.Guide('PopZ', WhereISPopZ_arr, NaNBoundary,
                                                 NucleoReg, location, PolymerLocl)
                        location_SpmX = np.array([SpmXLocl[0][0], SpmXLocl[1][0]])
                        SecondPermit = self.Guide('SpmX', WhereISSpmX_arr, NaNBoundary,
                                                  NucleoReg, location_SpmX, SpmXLocl)
                    else:
                        GuidePermit = self.Guide('PopZ', WhereISPopZ_arr, NaNBoundary,
                                                 NucleoReg, location, PolymerLocl)
                        SecondPermit = None

                    # Diffuse
                    DiffProbEDIT = DiffProb / (SelfSize ** (1 / 2))
                    if GuidePermit != 0 and random.random() < DiffProbEDIT:  ###DiffuProb is depended on size!!!!
                        ParticleMap = {'PopZ': WhereISPopZ_arr, 'SpmX': WhereISSpmX_arr}
                        ifinNucleo = 1
                        if len(SpmXLocl[0]) > 0:
                            PolymerLocl = {'PopZ': PolymerLocl, 'SpmX': SpmXLocl}
                            PopZ_movement, locl_for_bind = self.Diffuse(ParticleMap, GuidePermit,
                                                                        location, PolymerLocl, SecondPermit)
                            # Reshape coordinates for skipping method
                            if locl_for_bind != None:
                                tmp_locl_for_bind = np.append(locl_for_bind[0].T, locl_for_bind[1].T, axis=0)
                                # Update...
                                WhereISPopZ_arr = PopZ_movement[0]
                                # Update SpmX...
                                WhereISSpmX_arr = PopZ_movement[1]
                                # Check the location not in nucleoid reg
                                ifinNucleo = np.any(NucleoReg[locl_for_bind[0][0], locl_for_bind[0][1]])
                        else:
                            PopZ_movement, locl_for_bind = self.Diffuse(WhereISPopZ_arr, GuidePermit,
                                                                        location, PolymerLocl)
                            if locl_for_bind is not None:
                                # Reshape coordinates for skipping method
                                tmp_locl_for_bind = locl_for_bind.T
                                # Update...
                                WhereISPopZ_arr = PopZ_movement
                                # Check the location not in nucleoid reg
                                ifinNucleo = np.any(NucleoReg[locl_for_bind[0], locl_for_bind[1]])

                        if locl_for_bind is not None:
                            # Record coordinates for skipping method
                            tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in tmp_locl_for_bind]
                            skip_array = skip_array + tmplist

                        # Binding occurs when this molecule is not inside the nucleoid region...
                        if ifinNucleo == 0:
                            if len(SpmXLocl[0]) > 0:
                                ParticleMap = {'PopZ': WhereISPopZ_arr, 'SpmX': WhereISSpmX_arr}
                                LoclforBindDict = {'PopZ': locl_for_bind[0], 'SpmX': locl_for_bind[1]}
                                HowManyBindingTar, locl_candidate, \
                                SpmX_linked_candidate = self.interact('HybridPopZ', BindProb, ParticleMap,
                                                                      NucleoReg, LoclforBindDict, Cooperativity,
                                                                      SelfSize)
                                HowManyBindingTar_SpmX, locl_candidate_SpmX, \
                                _ = self.interact('HybridSpmX', BindProb, WhereISPopZ_arr,
                                                  NucleoReg, LoclforBindDict, Cooperativity, SelfSize)
                                HowManyBindingTar = HowManyBindingTar + HowManyBindingTar_SpmX
                                locl_candidate = np.append(locl_candidate, locl_candidate_SpmX, axis=1)
                                # location!????????????
                            else:
                                HowManyBindingTar, locl_candidate, \
                                SpmX_linked_candidate = self.interact('PopZ', BindProb, ParticleMap,
                                                                      NucleoReg, locl_for_bind, Cooperativity,
                                                                      SelfSize)

                            if HowManyBindingTar != 0:
                                if len(SpmXLocl[0]) > 0:
                                    PopZ_assemble, locl_for_skip = self.Bindgo('PopZ', WhereISPopZ_arr,
                                                                               locl_candidate, LoclforBindDict,
                                                                               WhereISSpmX_arr,
                                                                               SpmX_linked_candidate)
                                else:
                                    PopZ_assemble, locl_for_skip = self.Bindgo('PopZ', WhereISPopZ_arr,
                                                                               locl_candidate, locl_for_bind,
                                                                               WhereISSpmX_arr,
                                                                               SpmX_linked_candidate)
                                # Record coordinates for skipping method
                                tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in locl_for_skip]
                                skip_array = skip_array + tmplist
                                # Update...
                                WhereISPopZ_arr = PopZ_assemble[0]
                                WhereISSpmX_arr = PopZ_assemble[1]

            # Run SpmX diffusion
            elif SpmXgo == 1:  # Execute the behavior in membrane region while it's the turn in this grid in schedule.
                SerialNumber = WhereISSpmX_arr[coord[0], coord[1]]
                location = np.array([coord[0], coord[1]])
                # SpmX without binding with PopZ
                if SerialNumber == 1:  # Execute behavior only when the grid is occupied.
                    # Guide &?Diffuse
                    DiffuseSpmX, SpmX_for_bind, SpmX_skip = self.SingleSpmXGuide(WhereISSpmX_arr, coord)
                    if len(SpmX_skip)>0:
                        # Record coordinates for skipping method
                        tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in SpmX_skip]
                        skip_array = skip_array + tmplist
                        # Update
                        WhereISSpmX_arr = DiffuseSpmX

                        if len(SpmX_for_bind) > 0:
                            # Find binding targets
                            HowManyBindingTar, locl_candidate, _ = self.interact('SpmX', 0.5, WhereISPopZ_arr,
                                                                                 NucleoReg, SpmX_for_bind,
                                                                                 Cooperativity, SelfSize=1)

                            # Bind
                            if HowManyBindingTar != 0:
                                SpmX_linked_candidate = np.array([])
                                SpmX_assemble, locl_for_skip = self.Bindgo('SpmX', WhereISPopZ_arr,
                                                                           locl_candidate, SpmX_for_bind,
                                                                           WhereISSpmX_arr,
                                                                           SpmX_linked_candidate)
                                # Record coordinates for skipping method
                                tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in locl_for_skip]
                                skip_array = skip_array + tmplist
                                # Update...
                                WhereISPopZ_arr = SpmX_assemble[0]
                                WhereISSpmX_arr = SpmX_assemble[1]


                            # SpmX which binds with PopZ
                elif SerialNumber > 1 and np.isnan(SerialNumber) == 0:
                    # Find members
                    SpmXLocl = np.nonzero(WhereISSpmX_arr == SerialNumber)
                    PopZLocl = np.nonzero(WhereISPopZ_arr == SerialNumber)
                    SelfSize = len(PopZLocl[0]) + len(SpmXLocl[0])
                    # Guide
                    SpmXToward = self.Guide('SpmX', WhereISSpmX_arr, NaNBoundary, NucleoReg, location, SpmXLocl)
                    PopZToward = self.Guide('PopZ', WhereISPopZ_arr, NaNBoundary, NucleoReg, location, PopZLocl)
                    # Diffuse
                    DiffProbEDIT = DiffProb / (SelfSize ** (1 / 2))
                    if SpmXToward != 0 and \
                                    PopZToward != 0 and \
                                    random.random() < DiffProbEDIT:  ###DiffuProb is depended on size!!!!
                        ifinNucleo = 1
                        ParticleMap = {'PopZ': WhereISPopZ_arr, 'SpmX': WhereISSpmX_arr}
                        PolymerLocl = {'PopZ': PopZLocl, 'SpmX': SpmXLocl}
                        Particle_movement, Particle_for_bind = self.Diffuse(ParticleMap, PopZToward,
                                                                            location, PolymerLocl=PolymerLocl,
                                                                            SecondPermit=SpmXToward)

                        if Particle_for_bind is not None:
                            # Record coordinates for skipping method
                            tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in Particle_for_bind]
                            skip_array = skip_array + tmplist
                            # Update...
                            WhereISPopZ_arr = Particle_movement[0]
                            WhereISSpmX_arr = Particle_movement[1]
                            # Check the location not in nucleoid reg
                            ifinNucleo = np.any(NucleoReg[Particle_for_bind[0][0], Particle_for_bind[0][1]])

                        # Binding occurs when this molecule is not inside the nucleoid region...
                        if ifinNucleo == 0:
                            # Find binding targets
                            ParticleMap = {'SpmX': WhereISSpmX_arr, 'PopZ': WhereISPopZ_arr}
                            LoclforBindDict = {'PopZ': Particle_for_bind[0], 'SpmX': Particle_for_bind[1]}
                            HowManyBindingTar1, locl_candidate1, _ = self.interact('HybridSpmX', 0.5, WhereISPopZ_arr,
                                                                                   NucleoReg, LoclforBindDict,
                                                                                   Cooperativity, SelfSize=SelfSize)
                            HowManyBindingTar2, locl_candidate2, \
                            SpmX_linked_candidate = self.interact('HybridPopZ', BindProb, ParticleMap,
                                                                  NucleoReg, LoclforBindDict,
                                                                  Cooperativity, SelfSize=SelfSize)
                            HowManyBindingTar = HowManyBindingTar1 + HowManyBindingTar2
                            locl_candidate = np.append(locl_candidate1, locl_candidate2, axis=1)
                            # Bind
                            if HowManyBindingTar > 0:
                                Particle_assemble, locl_for_skip = self.Bindgo('HybridSpmX', WhereISPopZ_arr,
                                                                               locl_candidate, LoclforBindDict,
                                                                               WhereISSpmX_arr,
                                                                               SpmX_linked_candidate)

                                # Record coordinates for skipping method
                                tmplist = ['{0},{1}'.format(arr[0], arr[1]) for arr in locl_for_skip]
                                skip_array = skip_array + tmplist
                                # Update...
                                WhereISPopZ_arr = Particle_assemble[0]
                                WhereISSpmX_arr = Particle_assemble[1]

        return WhereISPopZ_arr, WhereISSpmX_arr

    def GetNeighiborEmpty(self, Species, particleMap, NaNBoundary, NucleoReg, location, PolymerLocl=None):
        """
        :param WhereISPopZ_csr: WhereISPopZ_csr
        :param NaNBoundary: NaNBoundary, indicate the space of cell
        :param NucleoReg: A region filled with chromosomal DNA is compacted
        :param location: represent the location which is scanned now
        :param PolymerLocl: if the grids scanned now are parts of multimers, "Polymerlocl" is not None and the location
        of polymers.
        """
        # initialize
        DiffuPermit = 0  # four-digit number
        NaN = NaNBoundary
        PopZ = particleMap
        Nucleo = NucleoReg
        Monitor = self.Monitor
        if Species == 'PopZ':
            NaN = NaNBoundary
            PopZ = particleMap
            Nucleo = NucleoReg
            Monitor = self.Monitor

        elif Species == 'SpmX':
            NaN = particleMap != np.nan
            PopZ = particleMap
            Nucleo = np.zeros((particleMap.shape[0], particleMap.shape[1]))
            Monitor = self.Monitor
            if len(PolymerLocl[0]) == 1:
                PolymerLocl = None

        row_Dir = np.array([-1, 1, 0, 0])
        col_Dir = np.array([0, 0, -1, 1])  # up down left right

        """
        If 'polymerlocl' is None, it indicates that there is a monomer in this grid.
        Following, we will sequentially check the neighbor grids.
        Once the grids are empty and not out of the boundary, molecules are allowed to move into.

        Otherwise, the information of location of polymer is picked and saved into 'polymerlocl' with an array type.
        We can scan by the row or column to know that there is any space for polymer diffusion.
        Importantly, using the difference between every two elements in a row or column information array is to
        aviod the gaps or holes in polymers.

        Diffusion-allowed region:
        thousands  hundreds  tens   ones
            0         0       0      0
            up       down    left  right
        index 0    index 1 index 2 index 3
        """
        if Nucleo[location[0], location[1]] == 1:  # Here is in nucleoid region.
            Permission = True
        else:
            Permission = Monitor(Nucleo, PopZ, location, 'Diffusion')

        if PolymerLocl == None:
            i = 3
            for r, c in zip(row_Dir, col_Dir):
                if PopZ[location[0] + r, location[1] + c] == 0 and NaN[
                            location[0] + r, location[1] + c] == 1:
                    if Nucleo[location[0] + r, location[1] + c] == 0:
                        DiffuPermit += 10 ** i
                    else:
                        if Permission is True:
                            DiffuPermit += 10 ** i
                i -= 1

        elif Species == 'SpmX':  # SpmX in Polymers
            Map = PopZ.copy()
            Map[PolymerLocl[0], PolymerLocl[1]] = 0
            Permit_arr = [1, 1, 1, 1]
            DiffuPermit = 0
            for locl_r, locl_c in zip(PolymerLocl[0], PolymerLocl[1]):
                i = 0
                for r, c in zip(row_Dir, col_Dir):
                    if Map[locl_r + r, locl_c + c] != 0 or NaN[
                                locl_r + r, locl_c + c] != 1:
                        Permit_arr[i] = 0
                    i += 1

            if np.any(Permit_arr):
                DiffuPermit = 1000 * Permit_arr[0] + 100 * Permit_arr[1] + 10 * Permit_arr[2] + Permit_arr[3]

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

                # UP
                if Nucleo[min(RowGrid) - 1, col] == 1:
                    if PopZ[min(RowGrid) - 1, col] != 0 or NaN[min(RowGrid) - 1, col] != 1:
                        UP = 0
                    elif Permission == False:
                        UP = 0
                else:
                    if PopZ[min(RowGrid) - 1, col] != 0 or NaN[min(RowGrid) - 1, col] != 1:
                        UP = 0

                # DOWN
                if Nucleo[max(RowGrid) + 1, col] == 1:
                    if PopZ[max(RowGrid) + 1, col] != 0 or NaN[max(RowGrid) + 1, col] != 1:
                        DOWN = 0
                    elif Permission == False:
                        DOWN = 0
                else:
                    if PopZ[max(RowGrid) + 1, col] != 0 or NaN[max(RowGrid) + 1, col] != 1:
                        DOWN = 0

                # inside UP & DOWN
                if UP == 1 or DOWN == 1:
                    if np.all(Continu == 1) == False:
                        flaw, = np.where(Continu != 1)
                        col = np.repeat(col, len(flaw))  # Make "col" as an array having the same length as "flaw"
                        # UP
                        if np.any(Nucleo[RowGrid[flaw + 1] - 1, col]) == 1:
                            if np.any(PopZ[RowGrid[flaw + 1] - 1, col]) != 0 or np.all(
                                    NaN[RowGrid[flaw + 1] - 1, col]) != 1:
                                UP = 0
                            elif Permission == False:
                                UP = 0
                        else:
                            if np.any(PopZ[RowGrid[flaw + 1] - 1, col]) != 0 or np.all(
                                    NaN[RowGrid[flaw + 1] - 1, col]) != 1:
                                UP = 0
                        # DOWN
                        if np.any(Nucleo[RowGrid[flaw] + 1, col]) == 1:
                            if np.any(PopZ[RowGrid[flaw] + 1, col]) != 0 or np.all(
                                    NaN[RowGrid[flaw] + 1, col]) != 1:
                                DOWN = 0
                            elif Permission == False:
                                DOWN = 0
                        else:
                            if np.any(PopZ[RowGrid[flaw] + 1, col]) != 0 or np.all(
                                    NaN[RowGrid[flaw] + 1, col]) != 1:
                                DOWN = 0

            # left&right
            UniqueRow = np.unique(PolymerLocl[0])
            for row in UniqueRow:
                ScanRow = np.where(PolymerLocl[0] == row)
                ColGrid = PolymerLocl[1][ScanRow]
                Continu = ColGrid[1:] - ColGrid[0:-1]

                # LEFT
                if Nucleo[row, min(ColGrid) - 1] == 1:
                    if PopZ[row, min(ColGrid) - 1] != 0 or NaN[row, min(ColGrid) - 1] != 1:
                        LEFT = 0
                    elif Permission == False:
                        LEFT = 0
                else:
                    if PopZ[row, min(ColGrid) - 1] != 0 or NaN[row, min(ColGrid) - 1] != 1:
                        LEFT = 0

                # RIGHT
                if Nucleo[row, max(ColGrid) + 1] == 1:
                    if PopZ[row, max(ColGrid) + 1] != 0 or NaN[row, max(ColGrid) + 1] != 1:
                        RIGHT = 0
                    elif Permission == False:
                        RIGHT = 0
                else:
                    if PopZ[row, max(ColGrid) + 1] != 0 or NaN[row, max(ColGrid) + 1] != 1:
                        RIGHT = 0

                # inside RIGHT & LEFT
                if LEFT == 1 or RIGHT == 1:
                    if np.all(Continu == 1) is False:
                        flaw, = np.where(Continu != 1)
                        row = np.repeat(row, len(flaw))  # Make "row" as an array having the same length as "flaw"
                        # LEFT
                        if np.any(Nucleo[row, ColGrid[flaw + 1] - 1]) == 1:
                            if np.any(PopZ[row, ColGrid[flaw + 1] - 1]) != 0 or np.all(
                                    NaN[row, ColGrid[flaw + 1] - 1]) != 1:
                                LEFT = 0
                            elif Permission == False:
                                LEFT = 0
                        else:
                            if np.any(PopZ[row, ColGrid[flaw + 1] - 1]) != 0 or np.all(
                                    NaN[row, ColGrid[flaw + 1] - 1]) != 1:
                                LEFT = 0
                        # RIGHT
                        if np.any(Nucleo[row, ColGrid[flaw] + 1]) == 1:
                            if np.any(PopZ[row, ColGrid[flaw] + 1]) != 0 or np.all(
                                    NaN[row, ColGrid[flaw] + 1]) != 1:
                                RIGHT = 0
                            elif Permission == False:
                                RIGHT = 0
                        else:
                            if np.any(PopZ[row, ColGrid[flaw] + 1]) != 0 or np.all(
                                    NaN[row, ColGrid[flaw] + 1]) != 1:
                                RIGHT = 0
            # Sum up the results
            DiffuPermit = UP * 1000 + DOWN * 100 + LEFT * 10 + RIGHT * 1

        return DiffuPermit

    def Diffugo(self, particleMap, DiffuPermit, location, PolymerLocl=None, SecondPermit=None):

        if isinstance(particleMap, dict):
            PopZ_arr = particleMap['PopZ']
            SpmX_arr = particleMap['SpmX']
            PolymerLocl_PopZ = PolymerLocl['PopZ']
            PolymerLocl_SpmX = PolymerLocl['SpmX']
            PolymerLocl = PolymerLocl_PopZ
        else:
            PopZ_arr = particleMap

        row_Dir = np.array([-1, 1, 0, 0])
        col_Dir = np.array([0, 0, -1, 1])  # up down left right

        if SecondPermit == None:
            DiffuPermit = np.array([int(i) for i in str(DiffuPermit)])  # Take diffusion permits apart to an array
            if len(DiffuPermit) != 4:
                for d in range(4 - len(DiffuPermit)):
                    DiffuPermit = np.append(0, DiffuPermit)
        else:
            DiffuPermit = np.array([int(i) for i in str(DiffuPermit)])  # Take diffusion permits apart to an array
            if len(DiffuPermit) != 4:
                for d in range(4 - len(DiffuPermit)):
                    DiffuPermit = np.append(0, DiffuPermit)
            Ori_permit = DiffuPermit.copy()

            SecondPermit = np.array([int(i) for i in str(SecondPermit)])  # Take diffusion permits apart to an array
            if len(SecondPermit) != 4:
                for d in range(4 - len(SecondPermit)):
                    SecondPermit = np.append(0, SecondPermit)

            CombinPermit = np.array([vi * vj for vi, vj in zip(DiffuPermit, SecondPermit)])
            DiffuPermit = CombinPermit

        DiffuChoose, = np.nonzero(DiffuPermit)  # 0: up, 1: down, 2: left, right: 3
        if len(DiffuChoose) == 0:
            BreakCtrl = 1
        else:
            BreakCtrl = 0
            randnum = np.random.randint(len(DiffuChoose))
            Radd = row_Dir[DiffuChoose[randnum]]
            Cadd = col_Dir[DiffuChoose[randnum]]

        if PolymerLocl is None and BreakCtrl == 0:  # monomer diffusion
            PopZ_arr[location[0], location[1]] = 0
            PopZ_arr[location[0] + Radd, location[1] + Cadd] = 1  # bug here...out of bounds
            locl_for_binding = np.array([location[0] + Radd, location[1] + Cadd])  # 1D array

        elif PolymerLocl is not None and BreakCtrl == 0:  # polymer diffusion
            SaveValue = PopZ_arr[PolymerLocl[0][0], PolymerLocl[1][0]]
            PopZ_arr[PolymerLocl[0], PolymerLocl[1]] = 0  # Elimiate the location of polymer
            PopZ_arr[PolymerLocl[0] + Radd, PolymerLocl[1] + Cadd] = SaveValue
            locl_for_binding = np.array([PolymerLocl[0] + Radd, PolymerLocl[1] + Cadd])  # 1D array
            if isinstance(particleMap, dict):
                # SpmX diffusion
                SpmX_arr[PolymerLocl_SpmX[0], PolymerLocl_SpmX[1]] = 0
                SpmX_arr[PolymerLocl_SpmX[0] + Radd, PolymerLocl_SpmX[1] + Cadd] = SaveValue
                locl_for_binding_SpmX = np.array([PolymerLocl_SpmX[0] + Radd, PolymerLocl_SpmX[1] + Cadd])
                locl_for_binding = (locl_for_binding, locl_for_binding_SpmX)
                PopZ_arr = (PopZ_arr, SpmX_arr)
        else:
            locl_for_binding = None

        return PopZ_arr, locl_for_binding

    def GetNeighiborFilled(self, Species, BindingThreshold,
                           particleMap, NucleoReg, location, Cooperativity, SelfSize=1):
        """
        :param WhereISPopZ_csr: WhereISPopZ_csr
        :param NaNBoundary: NaNBoundary, indicate the space of cell
        :param NucleoReg:
        :param location: New location after diffusion
        :param PolymerLocl:
        """
        # For coding reason, conferring none value to variables avoids unreferenced situation.
        Nucleo = None
        SpmX = None
        SpmX_Candidate_row = None
        SpmX_Candidate_col = None

        if Species == 'PopZ':
            SpmX_Candidate_row = np.array([])
            SpmX_Candidate_col = np.array([])
            PopZ = particleMap['PopZ']
            SpmX = np.uint8(particleMap['SpmX'])
            Nucleo = NucleoReg

        elif Species == 'SpmX':
            PopZ = np.uint8(particleMap.copy())
            Nucleo = np.zeros((particleMap.shape[0], particleMap.shape[1]))

        elif Species == 'HybridSpmX':
            # Use SpmX to find neighbor PopZ
            PopZ = particleMap.copy()
            Nucleo = np.zeros((PopZ.shape[0], PopZ.shape[1]))
            PopZ[location['PopZ'][0], location['PopZ'][1]] = 0
            location = location['SpmX']

        elif Species == 'HybridPopZ':
            # Use PopZ to find neighbor PopZ and SpmX
            SpmX_Candidate_row = np.array([])
            SpmX_Candidate_col = np.array([])
            SpmXori = np.uint8(particleMap['SpmX'].copy())
            PopZori = particleMap['PopZ']
            SpmX = SpmXori.copy()
            PopZ = PopZori.copy()
            SpmX[location['SpmX'][0], location['SpmX'][1]] = 0
            # SpmXLocl = np.nonzero(np.isnan(SpmX)==0)
            # PopZ[SpmXLocl[0], SpmXLocl[1]] = SpmX[SpmXLocl[0], SpmXLocl[1]]
            Nucleo = NucleoReg
            location = location['PopZ']

        Candidate_row = np.array([])  # Record the location of binding candidates
        Candidate_col = np.array([])
        row_Dir = np.array([-1, 1, 0, 0])
        col_Dir = np.array([0, 0, -1, 1])  # up down left right
        #
        size_record = np.array([])
        """
        If 'polymerlocl' is None, it indicates that there is a monomer in this grid.
        Following, we will sequentially check the neighbor grids.
        Once the grids are empty and not out of the boundary, molecules are allowed to move into.

        Otherwise, the information of location of polymer is picked and saved into 'polymerlocl' with an array type.
        We can scan by the row or column to know that there is any space for polymer diffusion.
        Importantly, using the difference between every two elements in a row or column information array is to
        aviod the gaps or holes in polymers.
        """

        if location.ndim == 1:
            for r, c in zip(row_Dir, col_Dir):
                Num = PopZ[location[0] + r, location[1] + c]
                Nuc = Nucleo[location[0] + r, location[1] + c]

                # Avoid repeated recording the info of the same candidate
                if Num != 0 and Nuc != 1:  ### ValueError: The truth value of an array with more than one element is ambiguous.
                    # Record row info of binding candidates
                    Candidate_row = np.append(Candidate_row, location[0] + r)
                    # Record column info of binding candidates
                    Candidate_col = np.append(Candidate_col, location[1] + c)
                    # Calculate the target size
                    if Num > 1:
                        tmpsize = np.count_nonzero(PopZ == Num)
                    else:
                        tmpsize = 1
                    # Record the size of binding candidates
                    size_record = np.append(size_record, tmpsize)

                if Species == 'PopZ' or Species == 'HybridPopZ':
                    if SpmX[location[0] + r, location[1] + c] != 0:
                        SpmX_Candidate_row = np.append(SpmX_Candidate_row, location[0] + r)
                        SpmX_Candidate_col = np.append(SpmX_Candidate_col, location[1] + c)

        elif Species == 'HybridSpmX':
            for locl_r, locl_c in zip(location[0], location[1]):
                for r, c in zip(row_Dir, col_Dir):
                    Num = PopZ[locl_r + r, locl_c + c]
                    Nuc = Nucleo[locl_r + r, locl_c + c]

                    # Avoid repeated recording the info of the same candidate
                    if Num != 0 and Nuc != 1:  ### ValueError: The truth value of an array with more than one element is ambiguous.
                        # Record row info of binding candidates
                        Candidate_row = np.append(Candidate_row, locl_r + r)
                        # Record column info of binding candidates
                        Candidate_col = np.append(Candidate_col, locl_c + c)
                        # Calculate the target size
                        if Num > 1:
                            tmpsize = np.count_nonzero(PopZ == Num)
                        else:
                            tmpsize = 1
                        # Record the size of binding candidates
                        size_record = np.append(size_record, tmpsize)

        elif location.ndim >= 2:  # HybridSpmX must be a polymer linked with PopZ
            # updown
            UniqueCol = np.unique(location[1])
            for col in UniqueCol:  # Scan one column in every loop
                ScanCol = np.where(location[1] == col)  # Find the grids in the same column in this multimer
                RowGrid = location[0][ScanCol]  # Find the row indices corresponding to the column indices
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
                        # Calculate the target size
                        if NumUP > 1:
                            tmpsize = np.count_nonzero(PopZ == NumUP)
                        else:
                            tmpsize = 1
                        # Record the size of binding candidates
                        size_record = np.append(size_record, tmpsize)

                    if Species == 'PopZ' or Species == 'HybridPopZ':
                        if SpmX[min(RowGrid) - 1, col] != 0:
                            SpmX_Candidate_row = np.append(SpmX_Candidate_row, (min(RowGrid) - 1))
                            SpmX_Candidate_col = np.append(SpmX_Candidate_col, col)

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
                        # Calculate the target size
                        if NumDOWN > 1:
                            tmpsize = np.count_nonzero(PopZ == NumDOWN)
                        else:
                            tmpsize = 1
                        # Record the size of binding candidates
                        size_record = np.append(size_record, tmpsize)

                    if Species == 'PopZ' or Species == 'HybridPopZ':
                        if SpmX[max(RowGrid) + 1, col] != 0:
                            SpmX_Candidate_row = np.append(SpmX_Candidate_row, (max(RowGrid) + 1))
                            SpmX_Candidate_col = np.append(SpmX_Candidate_col, col)

                # gaps
                if np.all(Continu == 1) == False:
                    flaw, = np.where(Continu != 1)
                    for gap in flaw:  # There is only one gap between the two grids.
                        Num = PopZ[RowGrid[gap + 1] - 1, col]
                        if Num != 0 and Nucleo[RowGrid[gap + 1] - 1, col] != 1:
                            Candidate_row = np.append(Candidate_row, (RowGrid[gap + 1] - 1))
                            Candidate_col = np.append(Candidate_col, col)
                            # Calculate the target size
                            tmpsize = np.count_nonzero(PopZ == Num)
                            # Record the size of binding candidates
                            size_record = np.append(size_record, tmpsize)
                            if Continu[gap] > 2:
                                if PopZ[RowGrid[gap] + 1, col] != 0 and Nucleo[RowGrid[gap] + 1, col] == 0:
                                    Candidate_row = np.append(Candidate_row, (RowGrid[gap] + 1))
                                    Candidate_col = np.append(Candidate_col, col)
                                    # Calculate the target size
                                    if PopZ[RowGrid[gap] + 1, col] > 1:
                                        tmpsize = np.count_nonzero(PopZ == PopZ[RowGrid[gap] + 1, col])
                                    else:
                                        tmpsize = 1
                                    # Record the size of binding candidates
                                    size_record = np.append(size_record, tmpsize)

            # left&right
            UniqueRow = np.unique(location[0])
            for row in UniqueRow:
                ScanRow = np.where(location[0] == row)
                ColGrid = location[1][ScanRow]
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
                        # Calculate the target size
                        if NumLEFT > 1:
                            tmpsize = np.count_nonzero(PopZ == NumLEFT)
                        else:
                            tmpsize = 1
                        # Record the size of binding candidates
                        size_record = np.append(size_record, tmpsize)

                    if Species == 'PopZ' or Species == 'HybridPopZ':
                        if SpmX[row, min(ColGrid) - 1] != 0:
                            SpmX_Candidate_row = np.append(SpmX_Candidate_row, row)
                            SpmX_Candidate_col = np.append(SpmX_Candidate_col, (min(ColGrid) - 1))

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
                        # Calculate the target size
                        if NumRIGHT > 1:
                            tmpsize = np.count_nonzero(PopZ == NumRIGHT)
                        else:
                            tmpsize = 1
                        # Record the size of binding candidates
                        size_record = np.append(size_record, tmpsize)

                    if Species == 'PopZ' or Species == 'HybridPopZ':
                        if SpmX[row, min(ColGrid) + 1] != 0:
                            SpmX_Candidate_row = np.append(SpmX_Candidate_row, row)
                            SpmX_Candidate_col = np.append(SpmX_Candidate_col, (max(ColGrid) + 1))

                # gaps
                if np.all(Continu == 1) is False:
                    flaw, = np.where(Continu != 1)
                    for gap in flaw:
                        Num = PopZ[row, ColGrid[gap + 1] - 1]
                        if Num != 0 and Nucleo[row, ColGrid[gap + 1] - 1] == 0:
                            Candidate_row = np.append(Candidate_row, row)
                            Candidate_col = np.append(Candidate_col, ColGrid[gap + 1] - 1)
                            # Calculate the target size
                            tmpsize = np.count_nonzero(PopZ == Num)
                            # Record the size of binding candidates
                            size_record = np.append(size_record, tmpsize)
                            if Continu[gap] > 2:
                                if PopZ[row, ColGrid[gap] + 1] != 0 and Nucleo[row, ColGrid[gap] + 1] == 0:
                                    Candidate_row = np.append(Candidate_row, row)
                                    Candidate_col = np.append(Candidate_col, ColGrid[gap] + 1)
                                    # Calculate the target size
                                    if PopZ[row, ColGrid[gap] + 1] > 1:
                                        tmpsize = np.count_nonzero(PopZ == PopZ[row, ColGrid[gap] + 1])
                                    else:
                                        tmpsize = 1
                                    # Record the size of binding candidates
                                    size_record = np.append(size_record, tmpsize)

        Candidate_locl = np.array([Candidate_row, Candidate_col], dtype=int)  # array([[row], [col]])

        if len(Candidate_row) > 0:
            WaitforDelete = []
            size_effect = 1
            for index, TarSize in enumerate(size_record):
                if Cooperativity == 'Off':
                    affinity = 1
                else:
                    # affinity = 2*(SelfSize + TarSize)/20
                    size_effect = self.BindGrowth(size1=SelfSize, size2=TarSize)
                if random.random() > size_effect * BindingThreshold:
                    WaitforDelete.append(index)
            Candidate_locl = np.delete(Candidate_locl, WaitforDelete, axis=1)

        if Species == 'PopZ' or Species == 'HybridPopZ':
            SpmX_Candidate_locl = np.array([SpmX_Candidate_row, SpmX_Candidate_col], dtype=int)
            if len(SpmX_Candidate_row) > 0:
                WaitforDelete = []
                size_effect = self.BindGrowth(size1=SelfSize, size2=1)
                for index in range(len(SpmX_Candidate_row)):
                    if random.random() > 10 * size_effect * BindingThreshold:
                        WaitforDelete.append(index)
                SpmX_Candidate_locl = np.delete(SpmX_Candidate_locl, WaitforDelete, axis=1)
        else:
            SpmX_Candidate_locl = np.array([]).reshape(2, 0)

        HowManyBindingTar = Candidate_locl.shape[1] + SpmX_Candidate_locl.shape[1]

        return HowManyBindingTar, Candidate_locl, SpmX_Candidate_locl

    def Bindgo(self, Species, WhereISPopZ_arr, Candidate_locl,
               location, WhereISSpmX_arr, SpmX_Candidate_locl):

        locl_target = []  # To solve UnboundLocalError

        PopZ_arr = WhereISPopZ_arr
        SpmX_arr = WhereISSpmX_arr

        # For coding reason, conferring none value to variables avoids unreferenced situation.
        SerialNumofTar = np.array([])
        SerialNumofSpmX = np.array([])
        Collector_tmp_SpmX = np.array([])
        list_col_SpmX = np.array([])
        list_col = np.array([])
        Collector_tmp = np.array([])
        list_row_Left = np.array([])
        list_col_Left = np.array([])
        list_row_LeftS = np.array([])
        list_col_LeftS = np.array([])
        list_row = np.array([])
        list_row_SpmX = np.array([])
        if Species == 'PopZ' or 'HybridPopZ' or 'SpmX' or 'HybridSpmX' or 'monoPopZ':
            # All serial number of binding target
            if len(Candidate_locl) > 0:  # Not an empty array
                SerialNumofTar = np.array(PopZ_arr[Candidate_locl[0], Candidate_locl[1]])
                monoPopZ = np.all(SerialNumofTar == 1)
                ctrl_PopZ = 1
            else:  # Empty
                ctrl_PopZ = 0
                monoPopZ = 1
            if len(SpmX_Candidate_locl) > 0:  # Not an empty array
                SerialNumofSpmX = np.array(SpmX_arr[SpmX_Candidate_locl[0], SpmX_Candidate_locl[1]])
                monoSpmX = np.all(SerialNumofSpmX == 1)
                ctrl_SpmX = 1
            else:  # Empty
                ctrl_SpmX = 0
                monoSpmX = 1

            # monomer binding target
            if monoPopZ and monoSpmX:
                # self is monomer
                # if location.squeeze().ndim == 1:
                if Species == 'monoPopZ' or Species == 'SpmX':
                    locl_target = np.array([]).reshape(2, 0)
                    # Give this new multimer with a serial number which was discarded.
                    if len(self.DegCollector) > 0:
                        WaitingNum = min(self.DegCollector)
                        if Species == 'SpmX':
                            SpmX_arr[location[0], location[1]] = WaitingNum
                        else:
                            PopZ_arr[location[0], location[1]] = WaitingNum
                        if ctrl_PopZ == 1:
                            PopZ_arr[Candidate_locl[0], Candidate_locl[1]] = WaitingNum
                            locl_target = np.append(locl_target, Candidate_locl, axis=1)
                        if ctrl_SpmX == 1 and Species != 'SpmX':  # SpmX do not bind with each other
                            SpmX_arr[SpmX_Candidate_locl[0], SpmX_Candidate_locl[1]] = WaitingNum
                            locl_target = np.append(locl_target, SpmX_Candidate_locl, axis=1)
                        locl_target = np.column_stack((locl_target[0], locl_target[1]))
                        self.DegCollector = np.delete(self.DegCollector, 0)

                    # Give this new multimer with a new serial number.
                    else:
                        if Species == 'SpmX':
                            SpmX_arr[location[0], location[1]] = self.LastSerialNum + 1
                        else:
                            PopZ_arr[location[0], location[1]] = self.LastSerialNum + 1
                        if ctrl_PopZ == 1:
                            PopZ_arr[Candidate_locl[0], Candidate_locl[1]] = self.LastSerialNum + 1
                            locl_target = np.append(locl_target, Candidate_locl, axis=1)
                        if ctrl_SpmX == 1:
                            SpmX_arr[SpmX_Candidate_locl[0], SpmX_Candidate_locl[1]] = self.LastSerialNum + 1
                            locl_target = np.append(locl_target, SpmX_Candidate_locl, axis=1)
                        locl_target = np.column_stack((locl_target[0], locl_target[1]))
                        self.LastSerialNum += 1

                else:
                    # self is a polymer
                    # monomer binding target
                    locl_target = np.array([]).reshape(2, 0)
                    if Species == 'HybridSpmX':
                        # Give monomer serial number as local SpmX polymer
                        SN_tmp = SpmX_arr[location['SpmX'][0][0], location['SpmX'][1][0]]
                    else:
                        # Give monomer serial number as local PopZ or Hybrid polymer
                        if isinstance(location, dict):
                            SN_tmp = PopZ_arr[location['PopZ'][0][0], location['PopZ'][1][0]]
                        else:
                            SN_tmp = PopZ_arr[location[0][0], location[1][0]]
                    if ctrl_PopZ == 1:  # PopZ binding targets
                        PopZ_arr[Candidate_locl[0], Candidate_locl[1]] = SN_tmp
                        if Candidate_locl.ndim == 1:
                            Candidate_locl = Candidate_locl.reshape(2, 1)
                        locl_target = np.append(locl_target, Candidate_locl, axis=1)
                    if ctrl_SpmX == 1:  # SpmX binding targets
                        SpmX_arr[SpmX_Candidate_locl[0], SpmX_Candidate_locl[1]] = SN_tmp
                        if SpmX_Candidate_locl.ndim == 1:
                            Candidate_locl = SpmX_Candidate_locl.reshape(2, 1)
                        locl_target = np.append(locl_target, SpmX_Candidate_locl, axis=1)
                    locl_target = np.column_stack((locl_target[0], locl_target[1]))  # For skipping rule

            # Polymer binding target
            # Monomer or Polymer in this grid
            else:
                LeftSpmX_row, LeftSpmX_col = np.array([], dtype=int), np.array([], dtype=int)
                LeftPopZ_row, LeftPopZ_col = np.array([], dtype=int), np.array([], dtype=int)
                # PopZ
                if ctrl_PopZ == 1:
                    fordel, = np.where(SerialNumofTar == 1)
                    if len(fordel) != 0:  # There is at least one monomer as binding target.
                        list_row = Candidate_locl[0][fordel].astype(int)
                        list_col = Candidate_locl[1][fordel].astype(int)
                        SerialNumofTar = np.unique(np.delete(SerialNumofTar, fordel))  # Delete the serial number 1
                    else:
                        SerialNumofTar = np.unique(SerialNumofTar)
                        list_row, list_col = np.array([], dtype=int), np.array([], dtype=int)

                    if len(SerialNumofTar) > 0:
                        # Appending binding target's coordinates
                        Collector_tmp = np.array([])
                        for i in range(len(SerialNumofTar)):
                            # Put the discarded SN to DegCollector
                            Collector_tmp = np.append(Collector_tmp, SerialNumofTar[i])
                            # In every time, the first number of SerialNumofTar is used to be serial number.
                            tmp_row, tmp_col = np.where(PopZ_arr == SerialNumofTar[i])
                            list_row = np.append(list_row, tmp_row)
                            list_col = np.append(list_col, tmp_col)
                            # Add PopZ-linked SpmX into the array for assigning serial number
                            Left_tmp_row, Left_tmp_col = np.nonzero(SpmX_arr == SerialNumofTar[i])
                            LeftSpmX_row = np.append(LeftSpmX_row, Left_tmp_row)
                            LeftSpmX_col = np.append(LeftSpmX_col, Left_tmp_col)

                # SpmX
                if ctrl_SpmX == 1 and Species != 'SpmX':
                    fordel_SpmX, = np.where(SerialNumofSpmX == 1)
                    if len(fordel_SpmX) != 0:  # There is at least one monomer as binding target.
                        list_row_SpmX = SpmX_Candidate_locl[0][fordel_SpmX].astype(int)
                        list_col_SpmX = SpmX_Candidate_locl[1][fordel_SpmX].astype(int)
                        SerialNumofSpmX = np.unique(
                            np.delete(SerialNumofSpmX, fordel_SpmX))  # Delete the serial number 1
                    else:
                        SerialNumofSpmX = np.unique(SerialNumofSpmX)
                        list_row_SpmX, list_col_SpmX = np.array([], dtype=int), np.array([], dtype=int)

                    if len(SerialNumofSpmX) > 0:  # ??????????????
                        Collector_tmp_SpmX = np.array([])
                        # Appending binding target's coordinates
                        for i in range(len(SerialNumofSpmX)):
                            # Put the discarded SN to DegCollector
                            Collector_tmp_SpmX = np.append(Collector_tmp_SpmX, SerialNumofSpmX[i])
                            # In every time, the first number of SerialNumofSpmX is used to be serial number.
                            tmp_row, tmp_col = np.where(SpmX_arr == SerialNumofSpmX[i])
                            list_row_SpmX = np.append(list_row_SpmX, tmp_row)
                            list_col_SpmX = np.append(list_col_SpmX, tmp_col)
                            # Add SpmX-linked PopZ into the array for assigning serial number
                            Left_tmp_row, Left_tmp_col = np.nonzero(PopZ_arr == SerialNumofSpmX[i])
                            LeftPopZ_row = np.append(LeftPopZ_row, Left_tmp_row)
                            LeftPopZ_col = np.append(LeftPopZ_col, Left_tmp_col)

                # popz only:
                # --> popz poly
                # popz and spmx present:
                # --> all spmx mono, PopZ poly/ all popz mono, spmx poly/ popz, spmx poly
                # Choose a serial number...
                # All serial numbers of binding targets are going to be the same as the first SN.
                if len(list_row) == 0:  # There are SpmX binding targets.
                    SN_chosen = Collector_tmp_SpmX[0]
                    Collector_tmp_SpmX = np.delete(Collector_tmp_SpmX, 0)
                    SpmX_arr[list_row_SpmX, list_col_SpmX] = SN_chosen
                    if len(LeftPopZ_row) > 0:
                        PopZ_arr[LeftPopZ_row, LeftPopZ_col] = SN_chosen
                    # ...including local grid
                    if isinstance(location, dict):
                        self.DegCollector = np.append(self.DegCollector,
                                                      PopZ_arr[location['PopZ'][0], location['PopZ'][1]])
                        PopZ_arr[location['PopZ'][0], location['PopZ'][1]] = SN_chosen
                        SpmX_arr[location['SpmX'][0], location['SpmX'][1]] = SN_chosen
                    else:
                        self.DegCollector = np.append(self.DegCollector,
                                                      PopZ_arr[location[0], location[1]])
                        PopZ_arr[location[0], location[1]] = SN_chosen

                    self.DegCollector = np.append(self.DegCollector, Collector_tmp_SpmX)

                elif len(list_row_SpmX) == 0 or Species == 'SpmX':  # There are PopZ binding targets.
                    SN_chosen = Collector_tmp[0]
                    Collector_tmp = np.delete(Collector_tmp, 0)
                    PopZ_arr[list_row, list_col] = SN_chosen
                    if len(LeftSpmX_row) > 0:
                        SpmX_arr[LeftSpmX_row, LeftSpmX_col] = SN_chosen
                    # ...including local grid
                    if isinstance(location, dict):
                        self.DegCollector = np.append(self.DegCollector,
                                                      PopZ_arr[location['PopZ'][0], location['PopZ'][1]])
                        PopZ_arr[location['PopZ'][0], location['PopZ'][1]] = SN_chosen
                        SpmX_arr[location['SpmX'][0], location['SpmX'][1]] = SN_chosen
                    elif Species == 'SpmX':
                        SpmX_arr[location[0], location[1]] = SN_chosen
                    else:
                        self.DegCollector = np.append(self.DegCollector,
                                                      PopZ_arr[location[0], location[1]])
                        PopZ_arr[location[0], location[1]] = SN_chosen
                    self.DegCollector = np.append(self.DegCollector, Collector_tmp)

                else:
                    if len(Collector_tmp_SpmX) > 0:
                        SN_chosen = Collector_tmp_SpmX[0]
                        Collector_tmp_SpmX = np.delete(Collector_tmp_SpmX, 0)
                    else:
                        SN_chosen = Collector_tmp[0]
                        Collector_tmp = np.delete(Collector_tmp, 0)

                    SpmX_arr[list_row_SpmX, list_col_SpmX] = SN_chosen
                    PopZ_arr[list_row, list_col] = SN_chosen
                    if len(LeftPopZ_row) > 0:
                        PopZ_arr[LeftPopZ_row, LeftPopZ_col] = SN_chosen
                    if len(LeftSpmX_row) > 0:
                        SpmX_arr[LeftSpmX_row, LeftSpmX_col] = SN_chosen
                    # ...including local grid
                    if isinstance(location, dict):
                        self.DegCollector = np.append(self.DegCollector,
                                                      PopZ_arr[location['PopZ'][0], location['PopZ'][1]])
                        PopZ_arr[location['PopZ'][0], location['PopZ'][1]] = SN_chosen
                        SpmX_arr[location['SpmX'][0], location['SpmX'][1]] = SN_chosen
                    else:
                        self.DegCollector = np.append(self.DegCollector,
                                                      PopZ_arr[location[0], location[1]])
                        PopZ_arr[location[0], location[1]] = SN_chosen
                    self.DegCollector = np.append(self.DegCollector, Collector_tmp_SpmX)
                    self.DegCollector = np.append(self.DegCollector, Collector_tmp)

                # These coordinates are recorded and then skipped while scanning grids.
                row_skip = np.append(list_row, list_row_SpmX)
                row_skip = np.append(row_skip, list_row_Left)
                row_skip = np.append(row_skip, list_row_LeftS)
                col_skip = np.append(list_col, list_col_SpmX)
                col_skip = np.append(col_skip, list_col_Left)
                col_skip = np.append(col_skip, list_col_LeftS)
                locl_target = np.column_stack((row_skip, col_skip))

            # Avoid get repeated serial number from DegCollector
            self.DegCollector = np.unique(self.DegCollector)
            self.DegCollector = np.delete(self.DegCollector, np.nonzero(self.DegCollector == 1))

        ParticleMap = (PopZ_arr, SpmX_arr)

        return ParticleMap, locl_target

    def Monitor(self, NucleoReg, PopZMap, coord, Purpose):
        permission = False  # It means that the nucleoid region is full.
        tmpMap = PopZMap.copy()
        mark = np.where(NucleoReg == 0)[0]  # To find where are nucleoid region
        tmpMap[mark[0], mark[1]] = 0  # Discard the information of PopZ outside the nucleoid region
        PopZinNucleo = np.sum(tmpMap)  # It indicates how many PopZ are inside the nucleoid region.
        left_space = round(0.75 * np.sum(NucleoReg)) - PopZinNucleo  # left_space stores how many grids are still empty.
        if left_space > 0:
            if Purpose == 'Diffusion':
                SN = PopZMap[coord[0], coord[1]]
                getSize = len(np.nonzero(PopZMap == SN)[0])
                if left_space - getSize >= 0 and getSize < 5:
                    permission = True
            elif Purpose == 'Binding':
                permission = int(left_space)
            elif Purpose == 'Production':
                permission = int(left_space) > 0

        return permission

    def GetEdge(self, WhereISPopZ_arr, DegProb, PolymerLocl):

        row_Dir = np.array([-1, 1, 0, 0])
        col_Dir = np.array([0, 0, -1, 1])  # up down left right
        PopZ_arr = WhereISPopZ_arr

        if isinstance(PolymerLocl, dict):
            PopZ_Locl = PolymerLocl['PopZ']
            SpmX_Locl = PolymerLocl['SpmX']
            size = len(PopZ_Locl[0]) + len(SpmX_Locl[0])
            PolyPure = 0 * PopZ_arr.copy()
            PolyPure[PopZ_Locl[0], PopZ_Locl[1]] = 1
            PolyPure[SpmX_Locl[0], SpmX_Locl[1]] = 1

        else:
            PopZ_Locl = PolymerLocl
            size = len(PopZ_Locl[0])
            PolyPure = 0 * PopZ_arr.copy()
            PolyPure[PopZ_Locl[0], PopZ_Locl[1]] = 1

        Edge_Record = np.uint8(np.zeros(len(PopZ_Locl[0])))  # if the score is zeros, the grid is inside the polymer
        counter = 0
        for r, c in zip(PopZ_Locl[0], PopZ_Locl[1]):
            for rr, cc in zip(row_Dir, col_Dir):
                if PolyPure[r + rr, c + cc] == 0:
                    Edge_Record[counter] += 1
            counter += 1

        WhereNoDeg, = np.where(Edge_Record == 0)
        PopZ_Locl = np.delete(PopZ_Locl, WhereNoDeg, axis=1)
        Edge_Record = np.delete(Edge_Record, WhereNoDeg)

        DegFate = np.zeros(len(PopZ_Locl[0]))

        # size_effect = 10*(1)/(1+(size/13)**10)
        for ind, times in enumerate(Edge_Record):
            for _ in range(times):
                if random.random() < DegProb * self.DegDecay(size=size):
                    DegFate[ind] = 1
                    break

        WhereNoDeg, = np.where(DegFate == 0)
        dePolymerLocl = np.delete(PopZ_Locl, WhereNoDeg, axis=1)

        return dePolymerLocl

    def Degradego(self, WhereISPopZ_arr, DegCandidate, SerialNum, WhereISSpmX_arr):
        WhereISPopZ_arr[DegCandidate[0], DegCandidate[1]] = 0  # Execute degradation
        # Find the part still exists after degradation
        left_part = np.uint8(np.where(WhereISPopZ_arr == SerialNum))
        left_SpmX = np.uint8(np.where(WhereISSpmX_arr == SerialNum))
        if len(left_part[0]) == 1 and len(left_SpmX[0]) == 0:  # it left one particle
            WhereISPopZ_arr[left_part[0], left_part[1]] = 1  # The state of the grid is from polymer to monomer
            self.DegCollector = np.append(self.DegCollector, SerialNum)
            self.DegCollector = np.unique(self.DegCollector)

        elif len(left_part[0]) > 1 and len(left_SpmX[0]) == 0:
            WhereISPopZ_arr = self.ReAssignSN(left_part, WhereISPopZ_arr, SerialNum)

        elif len(left_part[0]) >= 1 and len(left_SpmX[0]) > 0:
            ParticleMap = self.ReAssignSN(left_part, WhereISPopZ_arr,
                                          SerialNum, left_SpmX, WhereISSpmX_arr)
            WhereISPopZ_arr = ParticleMap[0]
            WhereISSpmX_arr = ParticleMap[1]

        elif len(left_part[0]) == 0 and len(left_SpmX[0]) > 0:
            WhereISSpmX_arr[left_SpmX[0], left_SpmX[1]] = 1

        else:
            self.DegCollector = np.append(self.DegCollector, SerialNum)
            self.DegCollector = np.unique(self.DegCollector)

        return WhereISPopZ_arr, WhereISSpmX_arr

    def ReAssignSN(self, loclarr, WhereISPopZ_arr, SerialNum, loclSpmX_arr=None, WhereISSpmX_arr=None):
        if loclSpmX_arr != None:
            locl_combine = np.append(loclarr, loclSpmX_arr, axis=1)
            WhereISSpmX_arr[loclSpmX_arr[0], loclSpmX_arr[1]] = 1
            WhereISPopZ_arr[loclarr[0], loclarr[1]] = 1
            ParticleMap = (WhereISPopZ_arr, WhereISSpmX_arr)
        else:
            locl_combine = loclarr
            ParticleMap = WhereISPopZ_arr

        find_same = np.zeros(len(loclarr[0]), dtype=object)
        # step 1:
        for r, c, ind in zip(loclarr[0], loclarr[1], range(len(loclarr[0]))):
            extract_r, = np.nonzero(locl_combine[0] == r)
            extract_c, = np.nonzero(locl_combine[1] == c)
            SearchbyR = extract_r[np.where(np.abs(locl_combine[1][extract_r] - c) == 1)]
            SearchbyC = extract_c[np.where(np.abs(locl_combine[0][extract_c] - r) == 1)]
            find_same[ind] = np.unique(np.append(np.append(ind, SearchbyR[:]), SearchbyC[:]))

        # step 2:
        tag = 0
        while True:
            try:
                arr = find_same[tag]  # Screen every array from 0 to ~
            except:
                break
            ori_len = len(arr)
            fordel = []
            for pos in arr:
                for n in range(tag + 1, len(find_same)):
                    if np.any(find_same[n] == pos):
                        find_same[tag] = np.uint8(np.append(find_same[tag], find_same[n]))
                        fordel.append(n)
            find_same[tag] = np.unique(find_same[tag])
            find_same = np.delete(find_same, np.uint8(np.unique(fordel)))
            new_len = len(find_same[tag])
            check = new_len - ori_len
            if check == 0:
                tag += 1

        # step 3:
        if len(find_same) > 0:
            self.DegCollector = np.append(self.DegCollector, SerialNum)
            self.DegCollector = np.unique(self.DegCollector)
            for arr in find_same:
                if len(arr) == 1:
                    WhereISPopZ_arr[loclarr[0][arr], loclarr[1][arr]] = 1
                else:
                    if len(self.DegCollector) > 0:
                        give_value = min(self.DegCollector)
                        self.DegCollector = np.delete(self.DegCollector, 0)
                    else:
                        give_value = self.LastSerialNum + 1
                        self.LastSerialNum += 1
                    # Confer the serial number for grids
                    if loclSpmX_arr != None:
                        SpmXReassign = np.nonzero(arr > len(loclarr[0]) - 1)
                        if len(SpmXReassign) > 0:
                            WhereISSpmX_arr[locl_combine[0][arr][SpmXReassign],
                                            locl_combine[1][arr][SpmXReassign]] = give_value
                            arr_N = np.delete(arr, SpmXReassign)
                            WhereISPopZ_arr[locl_combine[0][arr_N], locl_combine[1][arr_N]] = give_value
                        else:
                            WhereISPopZ_arr[locl_combine[0][arr], locl_combine[1][arr]] = give_value
                        ParticleMap = (WhereISPopZ_arr, WhereISSpmX_arr)
                    else:
                        WhereISPopZ_arr[locl_combine[0][arr], locl_combine[1][arr]] = give_value
                        ParticleMap = WhereISPopZ_arr

        return ParticleMap

    def SingleSpmXGuide(self, WhereISSpmX_arr, coord):
        # SpmX without binding with PopZ
        # Execute behavior only when the grid is occupied.
        SpmX_for_bind = np.array([])
        Skip_arr = np.array([])
        # Check grids which are empty and legal to move SpmX into
        cross = np.array([-1, 0, 1])  # Generate a 3 x 3 region for diffusion.
        Neighbors = cartesian((cross, cross))
        MoveableToX, MoveableToY = [], []  # Create two list for saving diffusable coordinates.
        for Neighbor in Neighbors:
            if Neighbor[0] != 0 or Neighbor[1] != 0:
                if WhereISSpmX_arr[coord[0] + Neighbor[0], coord[1] + Neighbor[1]] == 0:
                    MoveableToX.append(coord[0] + Neighbor[0])
                    MoveableToY.append(coord[1] + Neighbor[1])

        # Run SpmX diffusion
        if MoveableToX:
            choose = random.randint(0, len(MoveableToX) - 1)
            WhereISSpmX_arr[MoveableToX[choose], MoveableToY[choose]] = 1
            WhereISSpmX_arr[coord[0], coord[1]] = 0
            Skip_arr = np.column_stack((MoveableToX[choose], MoveableToY[choose]))  ####!!!!!
            SpmX_for_bind = np.array([MoveableToX[choose], MoveableToY[choose]])

        return WhereISSpmX_arr, SpmX_for_bind, Skip_arr
