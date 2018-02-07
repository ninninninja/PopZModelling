"""
+----------------------------------------------------------------------------------+
|   This is a main function for PopZ behavior simulation.                          |
|   Class needed :                                                                 |
|   1) Using CellSpace is to generate a rod-shaped & empty cell                    |
|   2) Behaviors is to execute what molecules would do inside cell                 |
|   3) Parameters is to determine or do sampling and then get parameters           |
|                                                                                  |
|   Revise content:                                                                |
|Not ready 1) Polymerization and multimers diffusion in nucleoid are allowed but   |
|       has some limit.                                                            |
|   2) Except NaN boundary and nucleoid region definition array, other             |
|   information or calculations are stored with general 2d array.                  |
|   3) To optimise computing efficiency, I plan to:                                |
|       3.1) Get neighbor information by scipy.ndimage.morphology.binary_dilation  |
|                                                                                  |
|   Update time: 20170618                                                          |
+----------------------------------------------------------------------------------+

"""
import time
import numpy as np
from PopZModel_Func.CellSpace import CellSpaceSet
from PopZModel_Func.Parameters import ParameterSet
from PopZModel_Func.Behaviors import BehaveSet
from scipy.sparse import coo_matrix, find
from PopZModel_Func.AnalysisFunctions import Realtime_Display
from PopZModel_Func.AnalysisFunctions import Statistcal_Analysis
import multiprocessing as mp
from functools import partial
import _pickle as cPickle

print('initialize...')

def input_parameters():
    # Give some essential information
    cellsize = np.array([4, 2])
    cellsizenow = cellsize ####[BECAREFUL]Edit here!!########
    accommodation = 950 #950
    Growth_Rate = 0
    TimeScale_change_rate = 0
    simulation_time = 2001#1802.5 #sec
    time_interval = 1

    # The parameters for adjustment of behavior of SpmX
    Trial = 10
    SpmXSampleNum = 10
    SpmX_amount = np.array([100, 100])
    SpmXMode = 'linear scale'

    # Sampling of SpmX
    Stat = Statistcal_Analysis()
    amount_upper = np.array([SpmX_amount[0]])
    amount_lower = np.array([SpmX_amount[1]])
    ParaSpmX = Stat.ODS(amount_upper, amount_lower, SpmXSampleNum, SpmXMode)

    # The parameters of degradation decay and binding growth
    SizeThresSample = 10
    CooperSample = 1
    ParaSizeThres = np.array([7.8, 7.8])  # m
    SizeThresmode = 'linear scale'
    ParaCooper = np.array([3.8, 3.8])  # n
    Coopermode = 'linear scale'

    # Sampling
    HillSize = np.array([SizeThresSample, CooperSample])
    HillUpper = np.array([ParaSizeThres[0], ParaCooper[0]])
    HillLower = np.array([ParaSizeThres[1], ParaCooper[1]])
    HillMode = np.array([SizeThresmode, Coopermode])
    ParaHill = Stat.CRS(HillUpper, HillLower, HillSize, HillMode)

    # Read parameter set
    #ParaRules = Stat.read_para_set('para_set.pickle')

    # Range of parameters
    ParaSize = SizeThresSample*CooperSample
    ParaProd = np.array([-3.5, -3.5]) # 2.25*10**(-3)  #2.25*10**(-3)/3 np.array([-4.5, -3.5])
    Prodmode = 'log scale'
    ParaDeg = np.array([-3.5, -3.5])  # 7.5*10**(-5) + 2.8*10**(-4)  3.55*10**(-4) # 3.55*10**(-4)/3 np.array([-2.5, -1.5])
    Degmode = 'log scale'
    ParaBind = np.array([-0.72, -0.72])  # 0.5
    Bindmode = 'log scale'
    ParaDiff = np.array([1, 1])  # 0.9   #0.9/3
    Diffmode = 'line scale'

    # Sampling
    ParaUpper = np.array([ParaProd[0], ParaDeg[0], ParaBind[0], ParaDiff[0]])
    ParaLower = np.array([ParaProd[1], ParaDeg[1], ParaBind[1], ParaDiff[1]])
    ParaMode = np.array([Prodmode, Degmode, Bindmode, Diffmode])
    ParaRules = Stat.ODS(ParaUpper, ParaLower, ParaSize, ParaMode)
    # Concatenate the two parameters array into one
    ParaAll = np.concatenate((ParaRules, ParaHill, ParaSpmX), axis = 1)
    ParaAll = np.repeat(ParaAll, Trial, axis=0)
    print(ParaAll)

    # Make a storage with a dictionary for transporting parameters setting...
    para_dict = {}
    para_dict['All Parameters'] = ParaAll
    para_dict['Time changes'] = TimeScale_change_rate
    para_dict['Cell size'] = cellsize
    para_dict['Accommodation'] = accommodation
    para_dict['Simulation time'] = simulation_time
    para_dict['Interval time'] = time_interval

    return para_dict

def behavior_screen(m, n):
    # affinity_input
    beta1 = 10
    #m1 = 20
    #n1 = 4
    # shrink_input
    beta2 = 10
    #m2 = 13
    #n2 = 10
    #SizeDependentBind = 'ON'
    AffinityIncrease = lambda beta, size1, size2, m, n: \
        beta*((size1 + size2)/m)**n / (1 + ((size1 + size2)/m)**n)
    #SizeDependentDeg = 'ON'
    ShrinkTerm = lambda beta, size, m, n: \
        (beta*1 / (1 + (size/m)**n))
    AI = partial(AffinityIncrease, beta = beta1, m = m, n = n)
    ST = partial(ShrinkTerm, beta = beta2, m = m, n = n)
    CellBehave = BehaveSet(AI, ST)

    return CellBehave

def input_cellspace(cellsize, accommodation):
    # Drawing an empty cell space...
    CellSpace = CellSpaceSet(cellsize, accommodation)
    CellSpace.amplify()
    CellSpace.gridsetting()
    CellSpace.findNaN()
    SparCellDict = CellSpace.Sparse_Cell()
    # Keep the original empty cell for restarting simulation.
    NucleoReg = SparCellDict['NucleoReg']
    NaNBoundary = SparCellDict['NaNBoundary']
    Membrane = SparCellDict['Membrane'].copy()
    Doable = SparCellDict['DoableReg'].copy() # The grides with value 1 are going to be scheduled.
    PopZMap = SparCellDict['EmptyCell'].copy()  # it is used to record localization of popz.
    print(PopZMap.shape)

    # Put information into a dictionary...
    map_dict = {}
    map_dict['PopZ Map'] = PopZMap
    map_dict['Nucleoid Region'] = NucleoReg
    map_dict['Inside the cell'] = NaNBoundary
    map_dict['Func from CellSpace'] = CellSpace
    map_dict['Membrane Region'] = Membrane
    map_dict['Doable Region'] = Doable

    return map_dict

def main(Para, Doable, Membrane, PopZMap, NucleoReg, NaNBoundary, extra_set, FUNC_CellSpace):
    # Determining the parameters...
    ParaSample = np.array([Para[0], Para[1], Para[2], Para[3]])
    Parameters = ParameterSet(extra_set['Time changes'],
                              extra_set['Cell size'], extra_set['Cell size'], ParaSample)
    ProProb = Parameters.production_prob()
    DegProb = Parameters.degradation_prob()
    BindProb = Parameters.binding_prob()
    DiffProb = Parameters.diffusion_prob()
    print(ProProb, DegProb, BindProb, DiffProb, Para[4], Para[5], Para[6])

    # Set up something important...
    CellBehave = behavior_screen(Para[4], Para[5])
    NaN_arr = NaNBoundary.toarray()
    Nucleo_arr = NucleoReg.toarray()
    Membrane = Membrane.toarray()
    Doable = Doable
    t = 0
    threshold = np.array([ProProb, DegProb, BindProb, DiffProb])
    AmountofSpmX = Para[6]
    Timelapse_recorder = {}
#==============================================================================================================#
    PopZMap = PopZMap.toarray()
    SpmXMap = Membrane.copy()
    while t != extra_set['Simulation time']:  # 1800 s = 30 min => cell division
        if t == 0:
            SpmXMap_filled = CellBehave.generateSpmX(SpmXMap, AmountofSpmX)
            SpmXMap = SpmXMap_filled
        else:
            # Queue the grids for scanning
            QueRow, QueCol = FUNC_CellSpace.Scheduling(Doable)
            QueCoordinate = np.column_stack((QueRow, QueCol))   # 2D array like arr([[1,2],[1,3],[1,4]....])
            NewMap = CellBehave.Scanning(QueCoordinate, SpmXMap,
                                         PopZMap, threshold, NaN_arr, Nucleo_arr)
            PopZMap = NewMap[0]
            SpmXMap = NewMap[1]

        t += extra_set['Interval time'] # 7.5

        if t%(extra_set['Interval time']*100) == 0:
            print('t',t)
            Timelapse_recorder[t] = {'PopZ':find(PopZMap), 'SpmX':find(np.uint8(SpmXMap))}

    return Timelapse_recorder


def async_multicore(main_part, lowerlimit, upperlimit, paras):
    pool = mp.Pool()    # Open multiprocessing pool
    result = []
    para_save = np.array([[]]).reshape(0, 7)
    # do computation
    for i in range(lowerlimit, upperlimit):
        res = pool.apply_async(main_part, args=(paras[i],))
        result.append(res)
        para_reshape = paras[i].reshape(1, 7)
        para_save = np.concatenate((para_save, para_reshape), axis = 0)
    pool.close()
    pool.join()

    output_dict = {}
    output_dict['Results'] = result
    output_dict['Parameters'] = para_save

    return output_dict

def export_results(output_dict):
    res = output_dict['Results']
    para_save = output_dict['Parameters']
    obj = {}
    ind = 0
    for data, paras in zip(res, para_save):
        obj['Data No. {0}'.format(ind)] = {'Prod':paras[0], 'Deg':paras[1],
                                           'Bind':paras[2],'Diff':paras[3],
                                           'SizeThres':paras[4], 'Coop':paras[5],
                                           'SpmXAmount':paras[6], 'Data':data.get()}
        ind +=1
    with open('result_SpmX_sub.pickle', 'wb') as picklefile:
        cPickle.dump(obj, picklefile, True)

if __name__ == '__main__':
    cpus = mp.cpu_count()
    print('Opening {0} cpus for simulation...'.format(cpus))
    para_dict = input_parameters()
    print('Preparing parameter sets...')
    map_dict = input_cellspace(para_dict['Cell size'], para_dict['Accommodation'])
    print('Preparing simulation space...')
    main_partial = partial(
        main, Doable = map_dict['Doable Region'], Membrane = map_dict['Membrane Region'],
        PopZMap = map_dict['PopZ Map'], NucleoReg = map_dict['Nucleoid Region'],
        NaNBoundary = map_dict['Inside the cell'], extra_set = para_dict,
        FUNC_CellSpace = map_dict['Func from CellSpace'])
    print('Loading all simulations...')
    start_time = time.time()
    output_dict = async_multicore(main_partial, 0, para_dict['All Parameters'].shape[0],
                                  para_dict['All Parameters'])
    end_time = time.time()
    print('Finished all simulations with {0} sec'.format(end_time - start_time))
    print('Saving data...')
    export_results(output_dict)
    print('Finishing program')