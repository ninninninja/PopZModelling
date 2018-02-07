# Import packages
import _pickle as cPickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, find
import time
import matplotlib.lines as Line
from matplotlib.gridspec import GridSpec
import os
import shutil
import matplotlib.cm as cm
import matplotlib

def PolymerFilter(find_res, size_threshold):
    # belongs to the Method: FocusOnPolymer
    # Remove monomers and small multimers
    # Process data with scipy.sparse.find
    # ------------------------------------
    # Create three empty array to save the results
    Row_res = np.uint8(np.array([]))
    Col_res = np.uint8(np.array([]))
    SN_res = np.uint8(np.array([]))
    
    # Get the serial numbers corresponding to theirs size
    SNarr, SizeArr = np.unique(find_res[2], return_counts = True)
    # Find the size of serial numbers are larger than the threshold
    KeepSN = SNarr[np.nonzero(SizeArr > size_threshold)]
    
    for SN in KeepSN:
        if SN == 1:
            continue
        AppendArr = np.where(find_res[2] == SN)
        Row_res = np.append(Row_res, find_res[0][AppendArr])
        Col_res = np.append(Col_res, find_res[1][AppendArr])
        SN_res = np.append(SN_res, find_res[2][AppendArr])

    return (Row_res, Col_res, SN_res, SizeArr, len(find_res[2]))


def PolymerScore(ProcessedArr):
    # belongs to the Method: FocusOnPolymer
    score = 0
    res = 0
    process_profile = np.zeros(86)
    col, counter = np.unique(ProcessedArr[1], return_counts = True)
    process_profile[col] = counter
    if len(counter)==0:
        res = 0
    else:
        # find if there is a large difference between biggest particle and the secondar one.
        _, sizearr = np.unique(ProcessedArr[2], return_counts = True)
        MaxSize = max(sizearr)
        sizearr[np.nonzero(sizearr==MaxSize)[0]] = 0
        allSize = np.sum(sizearr)
        if (ProcessedArr[4]-MaxSize) !=0:
            score = MaxSize / (ProcessedArr[4]-MaxSize)
        elif MaxSize == 0:
            score = 0
        else:
            score = MaxSize
        
        RightCount = np.count_nonzero(ProcessedArr[1] > 42)
        LeftCount = np.count_nonzero(ProcessedArr[1] < 43)

        if RightCount > 0 and LeftCount > 0:
            res = 2
        else:
            res = 1
        
    return score, res, process_profile, ProcessedArr[3]

# Analyzing formula for judging polar attribute
def Polarity_Analysis(Cell2darr, method):
    PolarProperties = {}
    if type(Cell2darr) == tuple:
            tmp2darr = np.zeros((44, 86))
            realarr = tmp2darr.copy()
            tmp2darr[Cell2darr[0], Cell2darr[1]] = 1
            realarr[Cell2darr[0], Cell2darr[1]] = Cell2darr[2]
            if np.any(np.isnan(Cell2darr[2])):
                print('nan error')
                exit()
        
    Cellshape = tmp2darr.shape
    CellLen = Cellshape[1]  # CellLen means the column length of 2D array.
    # To convert the values from serial number to bool, the purpose is that we just wonder the distribution of PopZ.
    Celltmp = tmp2darr!=0  # Convert data to bool type
    
    # The number of PopZ in the right of matrix is always bigger than the left.
    if CellLen % 2 == 0:
        SumL = np.sum(Celltmp[:, 0:round(CellLen / 2)-1])
        SumR = np.sum(Celltmp[:, round(CellLen / 2):])
    else:
        SumL = np.sum(Celltmp[:, 0:round(CellLen / 2)-1])
        SumR = np.sum(Celltmp[:, round(CellLen / 2)+1:])
    
    if SumR < SumL:
        Celltmp = np.fliplr(Celltmp)
        passValue = SumR
        SumR = SumL
        SumL = passValue
        
    ySumUp = np.zeros(CellLen)  # by which we can record PopZ profile along with x axis
    ySumUp = np.sum(Celltmp, axis=0)  # Sum up accroding to every column
    
    #
    Processed_profile = ySumUp.copy()  # Create a new variable for different processing
    PolarProperties['bool_data'] = Celltmp
    PolarProperties['SN_data'] = Cell2darr[2]
    PolarProperties['raw_profile'] = ySumUp
    PolarProperties['raw_data'] = realarr
#================Method: FocusOnPolymer==============================================================
    if method == 'FocusOnPolymer':
        ProcessedArr = PolymerFilter(Cell2darr, 12)
        Sig, res, Pprofile, Polydb = PolymerScore(ProcessedArr)
        PolarProperties['Definition'] = res
        PolarProperties['Score'] = Sig
        PolarProperties['processed_profile'] = Pprofile
        PolarProperties['polymer_distribution'] = Polydb
        
    else:
#================Method: Median=================================================================
        if method == 'median':
            if np.max(Processed_profile) < Cellshape[0]/5:  # It means the total number of PopZ is too less to see.
                PolarProperties['Definition'] = 0
                Processed_profile = Processed_profile*0
            elif np.min(Processed_profile) > Cellshape[0]/5:  # It means there are full of PopZ in everywhere.
                PolarProperties['Definition'] = 2
                Processed_profile = Processed_profile*0
            else:
                MidValue = np.median(ySumUp)
                for ind in range(len(ySumUp)):
                    if Processed_profile[ind] < 2 * MidValue:
                        Processed_profile[ind] = 0
                if CellLen % 2 == 0:        
                    ySumUp_L = Processed_profile[0:round(CellLen / 2)-1]
                    ySumUp_R = Processed_profile[round(len(ySumUp) / 2):]
                else:
                    ySumUp_L = Processed_profile[0:round(CellLen / 2)-1]
                    ySumUp_R = Processed_profile[round(len(ySumUp) / 2)+1:]
        
                if np.sum(ySumUp_R) > 0 and np.sum(ySumUp_L) > 0:
                    PolarProperties['Definition'] = 2
                elif np.sum(ySumUp_R) == 0 and np.sum(ySumUp_L) == 0:
                    PolarProperties['Definition'] = 0
                else:
                    PolarProperties['Definition'] = 1
                
            PolarProperties['processed_profile'] = Processed_profile

#================Method: LRratio=================================================================

        elif method == 'LRratio':
            tmpSum = ySumUp.copy()
            #
            if CellLen % 2 == 0:
                SumL = np.sum(tmpSum[0:round(CellLen / 2)-1])
                SumR = np.sum(tmpSum[round(CellLen / 2):])
            else:
                SumL = np.sum(tmpSum[0:round(CellLen / 2)-1])
                SumR = np.sum(tmpSum[round(CellLen / 2)+1:])
            #    
            if SumR < Cellshape[0]*Cellshape[1]/50 and SumL < Cellshape[0]*Cellshape[1]/50:
                PolarProperties['Definition'] = 0
            elif SumR / SumL > 2:
                PolarProperties['Definition'] = 1
            else:
                PolarProperties['Definition'] = 2
        
            PolarProperties['processed_profile'] = tmpSum

#================Method: ContiuousPeak===============================================================

        elif method == 'ContiuousPeak':
            if np.max(Processed_profile) < Cellshape[0]/5:  # It means the total number of PopZ is too less to see.
                PolarProperties['Definition'] = 0
                Processed_profile = Processed_profile*0
            elif np.min(Processed_profile) > Cellshape[0]/5:  # It means there are full of PopZ in everywhere.
                PolarProperties['Definition'] = 2
                Processed_profile = Processed_profile*0
            else:
                Band_Filter, = np.where(ySumUp < 2*np.median(Processed_profile))
                Processed_profile[Band_Filter] = 0
            
                flank = np.array([0, 0])
                binary_profile = np.concatenate((flank, Processed_profile, flank))!=0
                for i in range(2, len(ySumUp)+2):
                    if binary_profile[i] == True:
                        if np.count_nonzero(binary_profile[i-2:i+3]) == 1:
                            binary_profile[i] = False
                binary_profile = np.delete(binary_profile, [0, 1, len(binary_profile)-2,
                                                            len(binary_profile)-1])
                continu_filter = np.flatnonzero(binary_profile==False)
                Processed_profile[continu_filter] = 0
            
                #
                if CellLen % 2 == 0:        
                    ySumUp_L = Processed_profile[0:round(CellLen / 2)-1]
                    ySumUp_R = Processed_profile[round(CellLen / 2):]
                else:
                    ySumUp_L = Processed_profile[0:round(CellLen / 2)-1]
                    ySumUp_R = Processed_profile[round(CellLen / 2)+1:]
        
                if np.sum(ySumUp_R) > 0 and np.sum(ySumUp_L) > 0:
                    PolarProperties['Definition'] = 2
                elif np.sum(ySumUp_R) == 0 and np.sum(ySumUp_L) == 0:
                    PolarProperties['Definition'] = 0
                else:
                    PolarProperties['Definition'] = 1
                
            PolarProperties['processed_profile'] = Processed_profile
        
#================Method: ContiuousPeak===============================================================
            
        
    return PolarProperties


# Read simulation results from pickle files.
def Read_Data(filename, data_type, method, SpmX):
    with open(filename, 'rb') as file:
        res = cPickle.load(file)
    #==============Create build-in type empty lists for processed data storage===============
    uni_ori, uni_bool, uni_profile, uni_processed = [], [], [], []
    bi_ori, bi_bool, bi_profile, bi_processed = [], [], [], []
    di_ori, di_bool, di_profile, di_processed = [], [], [], []
    uni_record, bi_record, di_record = [], [], []  # Record the patterns by the changes of time
    uni_db, bi_db, di_db = [], [], []  #　Record the distribution of polymers by the changes of time
    #===========Create numpy empty array for recording information for drawing scatter========
    color_label = np.zeros((len(res)))
    size_label = color_label.copy()
    x = np.zeros(len(res))
    y = x.copy()
    z = x.copy()
    #===========Create storage space for recording hill function parameter screening=========
    SizeThres_di, SizeThres_uni, SizeThres_bi = np.array([]), np.array([]), np.array([])
    Cooper_di, Cooper_uni, Cooper_bi = np.array([]), np.array([]), np.array([])
    #========================================================================================
    di_ProdSave = []
    uni_ProdSave = []
    bi_ProdSave = []
    di_BindSave, uni_BindSave, bi_BindSave = [], [], []
    di_DegSave, uni_DegSave, bi_DegSave = [], [], []
    #========================================================================================
    if SpmX == 'ON':
        di_SpmXnum, uni_SpmXnum, bi_SpmXnum = [], [], []
    #========================================================================================
    score_record_uni, score_record_bi, score_record_di = [], [], []
    stability_record_uni, stability_record_di, stability_record_bi = [], [], []
    #========================================================================================
    kymo1 = np.zeros(20, dtype = object)
    kymo2 = []
    kymo3 = []
    print('Start processing the data...')
    print('There are {} piece of data in .pickle.'.format(len(res)))
    #if data_type == 'MultiSnap':
    # create a variable to store the changes of polarity by time 
    Polarity_tracker = np.zeros((4, 20))
    # polarity/time  1  2  3  4  5......
    #       diffuse  n  n  n  n  n......
    #      unipolar  n  n  n  n  n......
    #       bipolar  n  n  n  n  n......
    for ind, key in enumerate(res):
        tmp_dict = res['{}'.format(key)]
        single_record = np.zeros((2, 20))
        time_record, value_record = [], []
        for moment in tmp_dict['Data']:
            if moment > 2000:
                break
            det = Polarity_Analysis(tmp_dict['Data'][moment], method)
            Polarity_tracker[det['Definition'], int((moment/100)-1)] += 1
            single_record[0, int((moment/100)-1)] = det['Definition']
            single_record[1, int((moment/100)-1)] = int(np.sum(det['raw_profile']))
            if ind == 829:
                kymo1[int((moment/100)-1)] = det['raw_data']
            if moment == 2000:
                bool_data = det['bool_data']
                processed_profile = det['processed_profile']
                raw_profile = det['raw_profile']
                polydb = det['polymer_distribution']
                ori_data = det['SN_data']
                Score = det['Score']

    #==============================================================================
        if ind==0:
            arrsize = [44, 86]
            uni_hotspot = np.zeros((arrsize[0], arrsize[1]), dtype = int)
            bi_hotspot = uni_hotspot.copy()
            di_hotspot = uni_hotspot.copy()
            count_uni, count_bi, count_di = 0, 0, 0
    #==============================================================================
        x[ind] = np.log10(tmp_dict['Prod'])
        y[ind] = np.log10(tmp_dict['Deg'])
        z[ind] = np.log10(tmp_dict['Bind'])
        
        #print('deg',y[0], 'prod',x[0])
        if single_record[0, -1] == 0:
            # Diffused pattern...
            # Some essential information for 3D scatter
            color_label[ind] = 0.09#'c'
            size_label[ind] = 5
            # Record data
            di_hotspot += bool_data  # 1: PopZ hotspot
            count_di += 1  # 2: frequency counter
            di_bool.append(bool_data)  # 3: PopZ occupied raw data
            di_profile.append(raw_profile)  # 4: Unprocessed profile
            di_processed.append(processed_profile)  # 5: Processed profile
            di_record.append(single_record)
            di_db.append(polydb)
            di_ori.append(ori_data)
            SizeThres_di = np.append(SizeThres_di, tmp_dict['SizeThres'])
            Cooper_di = np.append(Cooper_di, tmp_dict['Coop'])
            di_ProdSave.append(np.log10(tmp_dict['Prod']))
            di_BindSave.append(np.log10(tmp_dict['Bind']))
            di_DegSave.append(np.log10(tmp_dict['Deg']))
            score_record_di.append(Score)
            stability = 0
            flipArr = np.flip(single_record[0], axis = 0)
            for ele in flipArr:
                if ele != 1:
                    break
                else:
                    stability+=1
            stability_record_di.append(stability)
            if SpmX == 'ON':
                di_SpmXnum.append(tmp_dict['SpmXAmount'])
                
        elif single_record[0, -1] == 1:
            # Unipolar pattern...
            # Some essential information for 3D scatter
            color_label[ind] = 0.4#'r'
            size_label[ind] = 60
            # Record data
            uni_hotspot += bool_data  # 1: PopZ hotspot
            count_uni += 1  # 2: frequency counter
            uni_bool.append(bool_data)  # 3: PopZ occupied raw data
            uni_profile.append(raw_profile)  # 4: Unprocessed profile
            uni_processed.append(processed_profile)  # 5: Processed profile
            uni_record.append(single_record)
            uni_db.append(polydb)
            uni_ori.append(ori_data)
            SizeThres_uni = np.append(SizeThres_uni, tmp_dict['SizeThres'])
            Cooper_uni = np.append(Cooper_uni, tmp_dict['Coop'])
            uni_ProdSave.append(np.log10(tmp_dict['Prod']))
            uni_BindSave.append(np.log10(tmp_dict['Bind']))
            uni_DegSave.append(np.log10(tmp_dict['Deg']))
            score_record_uni.append(Score)
            stability = 0
            flipArr = np.flip(single_record[0], axis = 0)
            for ele in flipArr:
                if ele != 1:
                    break
                else:
                    stability+=1
            stability_record_uni.append(stability)
            
            #
            if z[ind]<-0.9922582 and z[ind]>-0.9822583:
                print(ind, x[ind], y[ind], z[ind])
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(bool_data)
                plt.show()
                
#            if tmp_dict['Coop']<8.589 and tmp_dict['Coop']>8.57:
            if tmp_dict['Coop']<2.43 and tmp_dict['Coop']>2.41:
                if tmp_dict['SizeThres']<4.639 and tmp_dict['SizeThres']>4.62:
                    print(ind, tmp_dict['Coop'], tmp_dict['SizeThres'])
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.imshow(bool_data)
                    plt.show()
            
            if SpmX == 'ON':
                uni_SpmXnum.append(tmp_dict['SpmXAmount'])
            
        elif single_record[0, -1] == 2:
            # Bipolar pattern...
            # Some essential information for 3D scatter
            color_label[ind] = 0.9#'b'
            size_label[ind] = 20
            # Record data
            bi_hotspot += bool_data  # 1: PopZ hotspot
            count_bi += 1   # 2: frequency counter
            bi_bool.append(bool_data)  # 3: PopZ occupied raw data
            bi_profile.append(raw_profile)  # 4: Unprocessed profile
            bi_processed.append(processed_profile)  # 5: Processed profile
            bi_record.append(single_record)
            bi_db.append(polydb)
            bi_ori.append(ori_data)
            SizeThres_bi = np.append(SizeThres_bi, tmp_dict['SizeThres'])
            Cooper_bi = np.append(Cooper_bi, tmp_dict['Coop'])
            bi_ProdSave.append(np.log10(tmp_dict['Prod']))
            bi_BindSave.append(np.log10(tmp_dict['Bind']))
            bi_DegSave.append(np.log10(tmp_dict['Deg']))
            score_record_bi.append(Score)
            stability = 0
            flipArr = np.flip(single_record[0], axis = 0)
            for ele in flipArr:
                if ele != 1:
                    break
                else:
                    stability+=1
            stability_record_bi.append(stability)
            if SpmX == 'ON':
                bi_SpmXnum.append(tmp_dict['SpmXAmount'])
        else:
            print('There is something wrong.')
            pass
    #=========================================================================================
    #print('prod:', np.max(x), np.min(x), 'deg:', np.max(y), np.min(y), 'bind:', np.max(z), np.min(z))
    #
    hist_tup_di, hist_tup_bi, hist_tup_uni = (SizeThres_di, Cooper_di), (SizeThres_bi, Cooper_bi), (SizeThres_uni, Cooper_uni)
    Prod_record = (di_ProdSave, uni_ProdSave, bi_ProdSave)
    Bind_record = (di_BindSave, uni_BindSave, bi_BindSave)
    Deg_record = (di_DegSave, uni_DegSave, bi_DegSave)
    if SpmX == 'ON':
        SpmX_record = (di_SpmXnum, uni_SpmXnum, bi_SpmXnum)
    else:
        SpmX_record = ()
    
    Score_dict = {'uni':(score_record_uni, stability_record_uni),
                  'bi':(score_record_bi, stability_record_bi),
                  'di':(score_record_di, stability_record_di)}
    hist2d_dict = {'Unipolar':hist_tup_uni, 'Diffused':hist_tup_di, 'Bipolar':hist_tup_bi,
                   'ProdTrial':Prod_record, 'BindTrial':Bind_record, 'DegTrial':Deg_record,
                   'SpmXamount':SpmX_record,
                   'Score':Score_dict}
    #
    datapoint = np.sum(Polarity_tracker, axis= 0)
    Polarity_tracker[3, :] = datapoint
    #
    uni_folder = {'ori_data':uni_ori, 'bool_data':uni_bool, 'profile':uni_profile, 'Pprofile':uni_processed, 'SingleChanges':uni_record, 'Distribution':uni_db}
    di_folder = {'ori_data':di_ori, 'bool_data':di_bool, 'profile':di_profile, 'Pprofile':di_processed, 'SingleChanges':di_record, 'Distribution':di_db}
    bi_folder = {'ori_data':bi_ori, 'bool_data':bi_bool, 'profile':bi_profile, 'Pprofile':bi_processed, 'SingleChanges':bi_record, 'Distribution':bi_db}
    profile_folder = {'Unipolar':uni_folder, 'Diffused':di_folder, 'Bipolar':bi_folder}
    #
    hotspot_folder = {'uni_counter':count_uni, 'uni_hotspot':uni_hotspot/count_uni,
                     'di_counter':count_di, 'di_hotspot':di_hotspot/count_di,
                     'bi_counter':count_bi, 'bi_hotspot':bi_hotspot/count_bi}
    #
    scatter_folder = {'size':len(res), 'x':x, 'y':y, 'z':z,
                'color_label':color_label, 'size_label':size_label}
    
    #
    data_out = {'profile_folder':profile_folder, 'hotspot_folder':hotspot_folder,
                'scatter_folder':scatter_folder, 'Polarity_tracker':Polarity_tracker, 'hist2d_dict':hist2d_dict}
    return data_out, kymo1



def find2dmatch(arr_in, arr2d):
    str_arr_in = []
    str_arr2d = []
    matches = []
    for arr in arr_in:
        str_arr_in.append('{0},{1}'.format(arr[0], arr[1]))
    for arr in arr2d:
        str_arr2d.append('{0},{1}'.format(arr[0], arr[1]))
    str_arr2d = np.array(str_arr2d)
    for ele in str_arr_in:
            ind, = np.nonzero(str_arr2d == ele)
            for num in ind:
                matches.append(num)

    return matches


# Random pick some data to check the analysis
def uni_check(analysis_res, pickind, para_combin):
    pic_folder = {}
    print('unicheck')
    size_recorder = np.array([])
    if isinstance(pickind, list):
        for ii, arr in enumerate(pickind):
            print('arr', arr)
            if ii==0:
                key = 'Unipolar'
            elif ii==1:
                key = 'Bipolar'
            else:
                key = 'Diffused'
            print('These are {}.'.format(key))
            
            for ind in arr:
                print('m:', para_combin[0][ind], 'n:', para_combin[1][ind])
                plt.style.use('ggplot')
                fig = plt.figure()
                gs = GridSpec(8, 8)
                ax1 = fig.add_subplot(gs[0:4, :])
                ax1.imshow(analysis_res[key]['bool_data'][ind])
                ax2 = fig.add_subplot(gs[4:6, :])
                ax2.plot(analysis_res[key]['profile'][ind])
                ax3 = fig.add_subplot(gs[6:8, :])
                ax3.plot(analysis_res[key]['Pprofile'][ind])
                fig.tight_layout()
                pic_folder['Unipolar_m_{0}_n_{1}'.format(para_combin[0][ind], para_combin[1][ind])] = fig
                plt.show()
#========================================================================================#
                fig1 = plt.figure()
                ax = fig1.add_subplot(111)
                ax.set_xlabel('time (*150)')
                ax.set_ylabel('PopZ number', color='b')
                ax.tick_params('y', colors='b')
                a=ax.plot(analysis_res[key]['SingleChanges'][ind][1,:], 'b')
                
                axtw = ax.twinx()
                axtw.set_ylabel('Polarity fluctuation', color='r')
                axtw.tick_params('y', colors='r')
                b=axtw.plot(analysis_res[key]['SingleChanges'][ind][0,:], 'r')
                pic_folder['Unipolar_stability_m_{0}_n_{1}'.format(para_combin[0][ind], para_combin[1][ind])] = fig1
                plt.show()
#========================================================================================#
                _, counts = np.unique(analysis_res['Unipolar']['ori_data'][ind], return_counts = True)
                #size_arr, size_counts = np.unique(counts, return_counts = True)
                size_recorder = np.append(size_recorder, counts)
            
    else:
        print('validation')
        arr = pickind
        print(arr)
        key = 'Unipolar'
        
        for ind in arr:
            print('m:', para_combin[0][ind], 'n:', para_combin[1][ind])
            plt.style.use('ggplot')
            fig = plt.figure()
            gs = GridSpec(8, 8)
            ax1 = fig.add_subplot(gs[0:4, :])
            ax1.imshow(analysis_res[key]['bool_data'][ind])
            ax2 = fig.add_subplot(gs[4:6, :])
            ax2.plot(analysis_res[key]['profile'][ind])
            ax3 = fig.add_subplot(gs[6:8, :])
            ax3.plot(analysis_res[key]['Pprofile'][ind])
            fig.tight_layout()
            pic_folder['Unipolar_m_{0}_n_{1}'.format(para_combin[0][ind], para_combin[1][ind])] = fig
            plt.show()
#========================================================================================#
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            ax.set_xlabel('time (*150)')
            ax.set_ylabel('PopZ number', color='b')
            ax.tick_params('y', colors='b')
            a=ax.plot(analysis_res[key]['SingleChanges'][ind][1,:], 'b')
                
            axtw = ax.twinx()
            axtw.set_ylabel('Polarity fluctuation', color='r')
            axtw.tick_params('y', colors='r')
            b=axtw.plot(analysis_res[key]['SingleChanges'][ind][0,:], 'r')
            pic_folder['Unipolar_stability_m_{0}_n_{1}'.format(para_combin[0][ind], para_combin[1][ind])] = fig1
            plt.show()
#========================================================================================#
            _, counts = np.unique(analysis_res['Unipolar']['ori_data'][ind], return_counts = True)
            #size_arr, size_counts = np.unique(counts, return_counts = True)
            size_recorder = np.append(size_recorder, counts)
    
    #size_sort, freq = np.unique(size_recorder, return_counts = True)
    #fig2 = plt.figure()
    #ax_distribution = fig2.add_subplot(111)
    #ax_distribution.set_xlabel('PopZ Signal')
    #ax_distribution.set_ylabel('Pixels')
    #ax_distribution.bar(size_sort, freq, width=1.0, alpha=0.8, facecolor = 'black')
    #new_coord = np.nonzero(size_sort>0)
    #ax_distribution.bar(size_sort[new_coord], (freq*size_sort)[new_coord], width=1.0, facecolor = 'purple')
    #ax_distribution.set_xlim(0, 220)
    #fig2.savefig('[morethan10]PopZ_Signal_distribution_unipolar.png')
    #plt.show()
                
    return pic_folder



def hist2dforHill(hill_dict):
    from sklearn.utils.extmath import cartesian
    
    for key in hill_dict:
        if key=='ProdTrial':
            print('diff', len(hill_dict[key][0]))
            print('uni', len(hill_dict[key][1]))
            print('bi', len(hill_dict[key][2]))
            allrecord_bind = np.concatenate((hill_dict['ProdTrial'][0],
                                             hill_dict['ProdTrial'][1],
                                             hill_dict['ProdTrial'][2]))
            basal_bind = np.unique(allrecord_bind)
            allrecord = np.concatenate((hill_dict[key][0], hill_dict[key][1], hill_dict[key][2]))
            basal_level = np.unique(allrecord)
            diffu = np.append(hill_dict[key][0], basal_level)
            unipo = np.append(hill_dict[key][1], basal_level)
            bipo = np.append(hill_dict[key][2], basal_level)
            
            diffu, times_di = np.unique(diffu, return_counts = True)
            unipo, times_uni = np.unique(unipo, return_counts = True)
            bipo, times_bi = np.unique(bipo, return_counts = True)
            
            uni_show_index, = np.nonzero(hill_dict[key][1]==unipo[np.nonzero(times_uni==np.max(times_uni))])
            bi_show_index, = np.nonzero(hill_dict[key][2]==bipo[np.nonzero(times_uni==np.max(times_uni))])
            di_show_index, = np.nonzero(hill_dict[key][0]==diffu[np.nonzero(times_uni==np.max(times_uni))])
            show_index = [uni_show_index, bi_show_index, di_show_index]
            print('show index', show_index, len(show_index[0])+len(show_index[1])+len(show_index[2]))
            print('show',unipo[np.nonzero(times_uni==np.max(times_uni))], np.max(times_uni))
            print('trouble', np.nonzero(hill_dict[key][1]==-1.13358887))
            
            #diffu, di_edge = np.histogram(diffu, bins = len(basal_level))
            #unipo, uni_edge = np.histogram(unipo, bins = len(basal_level))
            #bipo, bi_edge = np.histogram(bipo, bins = len(basal_level))
            
            #print('Bind:', basal_bind[times_uni==10], 'Prod:', basal_level[times_uni==10])
            #print('Bind:', basal_bind[times_uni>=9], 'Prod:', basal_level[times_uni>=9])
            
            plt.style.use('bmh')
            fig1d = plt.figure()
            ax = fig1d.add_subplot(111)
            edit_times = ((times_di-1)/30, (times_bi-1)/30, (times_uni-1)/30)
            # Draw the histogram with bars.
            bar_di = ax.bar(np.arange(len(basal_level)), times_di-1, width=0.8, alpha=0.8, facecolor = 'w')#'blue')
            bar_bi = ax.bar(np.arange(len(basal_level)), times_bi-1, width=0.8, alpha=0.8, facecolor = 'w', bottom=times_di-1)
            bar_uni = ax.bar(np.arange(len(basal_level)), times_uni-1, width=0.8, alpha=0.8, facecolor = 'red', bottom=np.array(times_di-1+times_bi-1))
            
            #ax.set_xticks(np.linspace(min(basal_level), max(basal_level), 6))
            ax.set_yticklabels(np.round(np.linspace(min(basal_level), max(basal_level), 6), 1))
            ax.set_xlabel('Production Probability (log10)')
            #ax.set_xlabel('Serial number of combination of parameters')
            ax.set_ylabel('Counts')
            #plt.legend([bar_di, bar_uni, bar_bi], ["Diffused", "Unipolar", "Bipolar"], loc = 2)
            plt.show()
            
            #=================================================================================
            
            plt.clf()
            plt.style.use('bmh')
            figline = plt.figure()
            axline = figline.add_subplot(111)
            error_di = (((edit_times[0])*(1-edit_times[0]))/30)**(1/2)
            error_uni = (((edit_times[2])*(1-edit_times[2]))/30)**(1/2)
            error_bi = (((edit_times[1])*(1-edit_times[1]))/30)**(1/2)
            
            #axline.fill_between(np.arange(len(basal_level)), edit_times[0]+error_di,
            #                    edit_times[0]-error_di, alpha = 0.2, color = 'blue')
            #axline.fill_between(np.arange(len(basal_level)), edit_times[1]+error_bi,
            #                    edit_times[1]-error_bi, alpha = 0.2, color = 'orange')
            axline.fill_between(np.arange(len(basal_level)), edit_times[2]+error_uni,
                                edit_times[2]-error_uni, alpha = 0.2, color = 'red')
            
            #line_di = axline.plot(np.arange(len(basal_level)), edit_times[0], color = 'blue', label="Diffused")
            #line_bi = axline.plot(np.arange(len(basal_level)), edit_times[1], color = 'orange', label="Bipolar")
            line_uni = axline.plot(np.arange(len(basal_level)), edit_times[2], color = 'red', label="Unipolar")
            

            axline.set_xticks(np.arange(6)*20)
            axline.set_xticklabels(np.around(np.linspace(min(basal_level), max(basal_level), 6), decimals=2))
            axline.set_xlabel('Production Probability (log10)')
            #ax.set_xlabel('Serial number of combination of parameters')
            axline.set_ylabel('Ratio')
            #plt.legend(loc = 2)
            plt.show()
            
        elif key=='DegTrial':
            pass
        elif key=='BindTrial':
            pass
        elif key == 'SpmXamount':
            if hill_dict[key]:
                allrecord = np.concatenate((hill_dict[key][0], hill_dict[key][1], hill_dict[key][2]))
                basal_level = np.unique(allrecord)
                diffu = np.append(hill_dict[key][0], basal_level)
                unipo = np.append(hill_dict[key][1], basal_level)
                bipo = np.append(hill_dict[key][2], basal_level)
            
                diffu, times_di = np.unique(diffu, return_counts = True)
                unipo, times_uni = np.unique(unipo, return_counts = True)
                bipo, times_bi = np.unique(bipo, return_counts = True)
            
                plt.style.use('bmh')
                fig1d = plt.figure()
                ax = fig1d.add_subplot(111)
                edit_times = ((times_di-1)/100, (times_bi-1)/100, (times_uni-1)/100)
                # Draw the histogram with bars.
                width = 0.25
                bar_di = ax.bar(np.arange(len(basal_level)), times_di-1, width=width, alpha=0.8, facecolor = 'blue')
                bar_bi = ax.bar(np.arange(len(basal_level))+width, times_bi-1, width=width, alpha=0.8, facecolor = 'orange')
                bar_uni = ax.bar(np.arange(len(basal_level))+2*width, times_uni-1, width=width, alpha=0.8, facecolor = 'red')
            
                #ax.set_xticklabels(np.linspace(min(basal_level), max(basal_level), 6))
                #ax.set_xlabel('Production Probability (log10)')
                #ax.set_xlabel('Serial number of combination of parameters')
                #ax.set_ylabel('Counts')
                plt.legend([bar_di, bar_uni, bar_bi], ["Diffused", "Unipolar", "Bipolar"], loc = 2)
                plt.show()
        
        elif key =='Score':
            x_ori = np.linspace(2, 12, 20)
            y_ori = np.linspace(1, 10, 20)
            X, Y = np.meshgrid(x_ori, y_ori)
            basal = cartesian((x_ori, y_ori))
            
            string_index2D = []
            string_index2D.append(['{0},{1}'.format(arr[0],arr[1]) for arr in basal])
            string_index2D = np.array(string_index2D).reshape(20,20)
            string_index2D = string_index2D.T
            find_index2D = []
            find_index2D.append(['{0},{1}'.format(a,b) for a,b in zip(hill_dict['Unipolar'][0],hill_dict['Unipolar'][1])])

                #
            find_diff, find_bipo = [], []
            find_diff.append(['{0},{1}'.format(a,b) for a,b in zip(hill_dict['Diffused'][0],hill_dict['Diffused'][1])])
            find_bipo.append(['{0},{1}'.format(a,b) for a,b in zip(hill_dict['Bipolar'][0],hill_dict['Bipolar'][1])])
        
                
            correspond = np.zeros((string_index2D.shape[0], string_index2D.shape[1]), dtype=object)
            AllScore_di = correspond.copy()
            AllScore_bi = correspond.copy()
            score_heatmap = correspond.copy()
            stable_heatmap = correspond.copy()
            score_heatmap_di = correspond.copy()
            stable_heatmap_di = correspond.copy()
            score_heatmap_bi = correspond.copy()
            stable_heatmap_bi = correspond.copy()
            for i, arr in enumerate(string_index2D):
                for j, ele in enumerate(arr):
                    correspond[i, j] = np.nonzero(np.array(find_index2D)==ele)
                    AllScore_di[i, j] = np.nonzero(np.array(find_diff)==ele)
                    AllScore_bi[i, j] = np.nonzero(np.array(find_bipo)==ele)
                
                #
            for i in range(correspond.shape[0]):
                for j in range(correspond.shape[1]):
                    if len(correspond[i, j][1])>0:
                        locl = correspond[i, j][1].astype(int)
                        avg_score = np.sum(np.array(hill_dict['Score']['uni'][0])[locl])
                        avg_stab = np.sum(np.array(hill_dict['Score']['uni'][1])[locl])
                        score_heatmap[i, j] = avg_score
                        stable_heatmap[i, j] = avg_stab
                            
                    if len(AllScore_di[i, j][1])>0:
                        locl = AllScore_di[i, j][1].astype(int)
                        avg_score = np.sum(np.array(hill_dict['Score']['di'][0])[locl])
                        avg_stab = np.sum(np.array(hill_dict['Score']['di'][1])[locl])
                        score_heatmap_di[i, j] = avg_score
                        stable_heatmap_di[i, j] = avg_stab
                            
                    if len(AllScore_bi[i, j][1])>0:
                        locl = AllScore_bi[i, j][1].astype(int)
                        avg_score = np.sum(np.array(hill_dict['Score']['bi'][0])[locl])
                        avg_stab = np.sum(np.array(hill_dict['Score']['bi'][1])[locl])
                        score_heatmap_bi[i, j] = avg_score
                        stable_heatmap_bi[i, j] = avg_stab
                        
            #score_heatmap = score_heatmap.astype(int)
            #stable_heatmap = stable_heatmap.astype(int)
            score_heatmap = (score_heatmap + stable_heatmap_bi + stable_heatmap_di)/10
            stable_heatmap = (score_heatmap + stable_heatmap_bi + stable_heatmap_di)/10
            # setting colors
            norm = matplotlib.colors.Normalize(np.min(stable_heatmap), np.max(stable_heatmap))
            m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
            m.set_array([])
            fcolors = m.to_rgba(stable_heatmap)
            #colors = cm.jet(np.around(stable_heatmap/np.max(stable_heatmap), decimals = 0))
            colors = cm.jet(np.ones((stable_heatmap.shape[0], stable_heatmap.shape[1])))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            # color bar
            im = ax.imshow(stable_heatmap.astype(int), cmap='jet')
            plt.clf()
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            
            # bar3d
            print('score_heatmap',score_heatmap)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #ax.plot_surface(X,Y,score_heatmap, rstride=2, cstride=1, facecolors=fcolors,
            #                vmin=np.min(stable_heatmap), vmax=np.max(stable_heatmap))
            ly = len(score_heatmap[0])
            lx = len(score_heatmap[:,0])
            xpos = np.arange(0,lx,1)    # Set up a mesh of positions
            ypos = np.arange(0,ly,1)
            xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)

            xpos = xpos.flatten()   # Convert positions to 1D array
            ypos = ypos.flatten()
            zpos = np.zeros(lx*ly)

            dx = 0.5 * np.ones_like(zpos)
            dy = dx.copy()
            dz = score_heatmap.flatten()

            colors = colors.flatten()
            colors = colors.reshape(int(len(colors)/4), 4)
            for i in range(len(colors)):
                ax.bar3d(xpos[i], ypos[i], zpos[i], dx, dy, dz[i], alpha=0.2, color=colors[i])

            # labels
            ax.set_xlabel('Size of threshold')
            ax.set_ylabel('Cooperativity')
            ax.set_zlabel('Signal to noise')
            ax.set_xticklabels(np.round(np.linspace(2, 12, 5), 2), minor=False)
            ax.set_yticklabels(np.round(np.linspace(1, 10, 5), 2), minor=False)

            plt.grid('on')
            plt.show()
                
            plt.pcolor(score_heatmap, cmap = 'viridis')
            plt.show()
            
        else:
            #print('bug', key)
            # Process data 
            folder = {}
            x_ori = np.linspace(2, 12, 20)
            y_ori = np.linspace(1, 10, 20)
            basal = cartesian((x_ori, y_ori))
            string_index2D = []
            string_index2D.append(['{0},{1}'.format(arr[0],arr[1]) for arr in basal])
            
            x, y = basal[:, 0], basal[:, 1]
            x, y = np.append(x, hill_dict[key][0]), np.append(y, hill_dict[key][1])
            #print(key, 'N,K:', hill_dict[key][1][0], hill_dict[key][0][0])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(x, bins = 20)
            plt.show()
            heatmap, xedges, yedges = np.histogram2d(x, y, bins = (20, 20))
            heatmap -= 1
        
            # Generate figure
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(key)
            im = ax.pcolor(heatmap.T, cmap = 'viridis')
            # Set labels
            ax.set_xticks(np.arange(heatmap.T.shape[0])+0.5, minor=False)
            ax.set_yticks(np.arange(heatmap.T.shape[1])+0.5, minor=False)
            ax.set_xticklabels(np.round(np.linspace(2, 12, 20), 2), minor=False)
            ax.set_yticklabels(np.round(np.linspace(1, 10, 20), 2), minor=False)
            ax.set_xlabel('Threshold of size')
            ax.set_ylabel('Coopertivity')
            plt.xticks(rotation=90)
            fig.colorbar(im, ax = ax)
            folder[key] = fig
            plt.show()
            
            # 
        
            if key == 'Unipolar':
                # Find where present the highest freq of unipolar.
                #coord = np.nonzero(heatmap.T==np.max(heatmap.T))
                coord = np.nonzero(heatmap.T==7)
                #print(coord)
                # Generate array recording the original inputs.
                arr_tar = np.column_stack((hill_dict[key][0], hill_dict[key][1]))
                # Flip the array which represents y axis corresponding to the fig
                y_tmp = np.flip(y_ori, axis = 0)
                # Get values from data plot.
                x_find, y_find = x_ori[coord[1]], y_ori[coord[0]]
                arr_in = np.column_stack((x_find, y_find))
                matches = find2dmatch(arr_in, arr_tar)
                print(matches)
                matches = np.unique(matches)
                print('matches', matches)
    
    return {'fig1d':fig1d, 'figline':figline}, show_index


# Random pick some data to check the analysis
def Random_check(analysis_res, picknum):
    pic_folder = {}
    for key in analysis_res:
        if len(analysis_res[key]['bool_data']) !=0:
            picked_ind = np.random.randint(0, len(analysis_res[key]['bool_data']), picknum)
            for ind in picked_ind:
                print(key, ind)
                MidValue = np.median(analysis_res[key]['profile'][ind])
                print('midvalue', MidValue)
                plt.style.use('bmh')
                fig = plt.figure()
                gs = GridSpec(8, 8)
                ax1 = fig.add_subplot(gs[0:4, :])
                ax1.imshow(analysis_res[key]['bool_data'][ind])
                ax2 = fig.add_subplot(gs[4:6, :])
                ax2.plot(analysis_res[key]['profile'][ind])
                ax3 = fig.add_subplot(gs[6:8, :])
                ax3.plot(analysis_res[key]['Pprofile'][ind])
                fig.tight_layout()
                pic_folder['NO{0}_{1}'.format(ind, key)] = fig
                plt.show()
#========================================================================================#
                fig1 = plt.figure()
                ax = fig1.add_subplot(111)
                ax.set_xlabel('time (*150)')
                ax.set_ylabel('PopZ number', color='b')
                ax.tick_params('y', colors='b')
                a=ax.plot(analysis_res[key]['SingleChanges'][ind][1,:], 'b')
                
                axtw = ax.twinx()
                axtw.set_ylabel('Polarity fluctuation', color='r')
                axtw.tick_params('y', colors='r')
                b=axtw.plot(analysis_res[key]['SingleChanges'][ind][0,:], 'r')
                pic_folder['NO{0}_{1}_Steadiness'.format(ind, key)] = fig1
                plt.show()
#========================================================================================#
                # the histogram of the data
                fig2 = plt.figure()
                axdb = fig2.add_subplot(111)
                axdb.hist(analysis_res[key]['Distribution'][ind], bins = 50, normed=1, facecolor='green')
                axdb.set_xlabel('Size')
                axdb.set_ylabel('Amount')
                axdb.set_title('The distribution of size of PopZ polymers')
                axdb.grid(True)
                pic_folder['NO{0}_{1}_PolymerDistribution'.format(ind, key)] = fig2
                plt.show()
                
    return pic_folder


def KingChosen(arr2d, arr1d):
    if len(arr1d) % 2 == 0:        
        ySumUp_L = arr1d[0:round(len(arr1d) / 2)-1]
        ySumUp_R = arr1d[round(len(arr1d) / 2):]
    else:
        ySumUp_L = arr1d[0:round(len(arr1d) / 2)-1]
        ySumUp_R = arr1d[round(len(arr1d) / 2)+1:]
    LRratio = np.sum(ySumUp_R) / np.sum(ySumUp_L)
    _, size_recorder = np.unique(arr2d, return_counts = True)
    score = size_recorder
    noise = (np.sum(score)-max(score))#/(len(score)-1)
    if noise == 0:
        score_new = 80
    else:
        score_new = (max(score))/noise
    return score_new, max(score)#-np.percentile(score, 95, axis=0)



def Pole_background(analysis_res):
    score_recorder = {}
    for key in analysis_res:
        score_recorder[key] = {'score':[], 'pole':[], 'color':[]}
        for ind in range(len(analysis_res[key]['ori_data'])):
            score, polesize = KingChosen(analysis_res[key]['ori_data'][ind], analysis_res[key]['profile'][ind])
            score_recorder[key]['score'].append(score)
            score_recorder[key]['pole'].append(polesize)
            if key == "Unipolar":
                score_recorder[key]['color'].append('orange')
            elif key =="Bipolar":
                score_recorder[key]['color'].append('purple')
            else:
                score_recorder[key]['color'].append('blue')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter1_proxy = ax.scatter(score_recorder['Diffused']['score'], score_recorder['Diffused']['pole'], c = score_recorder['Diffused']['color'], alpha=0.5)
    scatter3_proxy = ax.scatter(score_recorder['Bipolar']['score'], score_recorder['Bipolar']['pole'], c = score_recorder['Bipolar']['color'], alpha=0.5)
    scatter2_proxy = ax.scatter(score_recorder['Unipolar']['score'], score_recorder['Unipolar']['pole'], c = score_recorder['Unipolar']['color'], alpha=0.5)
    ax.set_xlabel('Score')
    ax.set_ylabel('Pole size')
    ax.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy], ['Diffused', 'Unipolar', 'Bipolar'], numpoints = 1)
    plt.show()


# Drawing simulation result with 3d scatter plot
def Draw_3dScatter(input_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(input_dict['x'], input_dict['y'], input_dict['z'], cmap='hot', vmax=1, vmin=0,
               c=input_dict['color_label'], s=input_dict['size_label'])

    scatter1_proxy = Line.Line2D([0],[0], linestyle="none", c='black', marker = 'o')
    scatter2_proxy = Line.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
    scatter3_proxy = Line.Line2D([0],[0], linestyle="none", c='y', marker = 'o')
    ax.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy], ['label1', 'label2', 'label3'], numpoints = 1)
    ax.set_xlabel('Production')
    ax.set_ylabel('Degradation')
    ax.set_zlabel('Binding')
    fig.tight_layout()
    plt.show()
    return fig


def Pie_plot(size_arr):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Diffused', 'Unipolar', 'Bipolar'
    explode = (0, 0.1, 0)  # only "explode" the 2nd slice

    fig1, ax1 = plt.subplots()
    ax1.pie(size_arr, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    return fig1


def Hotspot_plot(input_dict):
    print('{} data points belong to diffused pattern.'.format(input_dict['di_counter']))
    print('{} data points belong to unipolar pattern.'.format(input_dict['uni_counter']))
    print('{} data points belong to bipolar pattern.'.format(input_dict['bi_counter']))
    PieFig = Pie_plot([input_dict['di_counter'], input_dict['uni_counter'], input_dict['bi_counter']])
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    ax1.imshow(input_dict['di_hotspot'], vmax=1, vmin=0)
    ax1.set_title('Diffused')
    ax2.imshow(input_dict['uni_hotspot'], vmax=1, vmin=0)
    ax2.set_title('Unipolar')
    ax3.imshow(input_dict['bi_hotspot'], vmax=1, vmin=0)
    ax3.set_title('Bipolar')
    fig.tight_layout()
    plt.show()
    return {'PopZ_Hotspot':fig, 'PopZ_PieChart':PieFig}


def Tracker_plot(arr):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    a, = ax.plot(arr[0, :], label='Diffuse')
    b, = ax.plot(arr[1, :], label='Unipolar')
    c, = ax.plot(arr[2, :], label='Bipolar')
    ax.set_xlabel('time (*150)')
    ax.set_ylabel('Number of patterns')
    plt.legend(handles=[a, b, c])
    plt.show()
    return fig


def kymograph(ori_data):
    for ind, arr in enumerate(ori_data):
        arr = np.rot90(arr)
        color = 0
        if ind==0:
            kymo = arr.copy()
            kymo[np.nonzero(arr!=0)] = 1
            sn, counts = np.unique(arr, return_counts=True)
            sn = np.delete(sn, [0, 1])
            counts = np.delete(counts, [0, 1])
            print(counts)
            if np.any(arr):
                pick = sn[np.nonzero(counts==max(counts))][0]
                color = 3
                kymo[np.nonzero(arr == pick)] = color
        else:
            sn, counts = np.unique(arr, return_counts=True)
            sn = np.delete(sn, [0, 1])
            counts = np.delete(counts, [0, 1])
            pick = sn[np.nonzero(counts==max(counts))][0]
            color = 3
            tmp = arr.copy()
            tmp[np.nonzero(arr!=0)] = 1
            tmp[np.nonzero(arr == pick)] = color
            kymo = np.concatenate((kymo, tmp), axis = 1)
    
    plt.style.use('ggplot')
    fig = plt.figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.imshow(kymo, cmap = cm.hot, vmax=5, vmin=0)
    plt.grid('off')
    
    ax = fig.add_subplot(211)
    ax.imshow(kymo!=0, cmap = cm.hot, vmax=2, vmin=0)
    plt.grid('off')
    plt.show()


def Auto_HousingData(filename, waitingfiles, methodname, rmSwitch):
    path = r'C:\Users\lynch\AppData\Local\Programs\Python\Python35\CodeOnIPython'
    os.chdir(path)
    folder_name = 'analysis_'+filename.split('.')[0]
    if os.path.exists(folder_name) and rmSwitch==1:
        shutil.rmtree(folder_name)
    try:
        os.mkdir(folder_name)
    except:
        print('The folder is exist')
    finally:
        os.chdir(folder_name)
    #
    lower_folder_name = 'method_{}'.format(methodname)
    if os.path.exists(lower_folder_name):
        shutil.rmtree(lower_folder_name)
    os.mkdir(lower_folder_name)
    os.chdir(lower_folder_name)
    # Saving files in corresponding folders
    # Data structure:
    # dictionary{'method1':[pic1, pic2, pic3], 'method2':[pic_jason1, pic_jason2],...}
    for pic in waitingfiles:
        if isinstance(waitingfiles[pic], dict) is False:
            waitingfiles[pic].savefig('{}.png'.format(pic))
        else:
            for picname in waitingfiles[pic]:
                waitingfiles[pic][picname].savefig('{}.png'.format(picname))


if __name__ == '__main__':
    path = r'C:\Users\lynch\AppData\Local\Programs\Python\Python35\CodeOnIPython'
    #methods = ['median', 'ContiuousPeak', 'LRratio']
    methods =['FocusOnPolymer'] #['LRratio']
    ctrl = 1
    for methodname in methods:
        os.chdir(path)
        filename = '[MonoEdit][HillScreen][deg3.5][bind0.8]result_SpmX_20170711.pickle'
        data_out, kymo1 = Read_Data(filename, 'Multisnap', methodname, 'OFF')
        #kymograph(kymo1)
        check_data_num = 20
        fig1d, show_index = hist2dforHill(data_out['hist2d_dict'])
        figbonus = uni_check(data_out['profile_folder'], show_index ,data_out['hist2d_dict']['Unipolar'])
        #fig1 = Draw_3dScatter(data_out['scatter_folder'])
        #fig2 = Hotspot_plot(data_out['hotspot_folder'])
        #fig3 = Random_check(data_out['profile_folder'], check_data_num)
        #fig4 = Tracker_plot(data_out['Polarity_tracker'])
        #figfolder = {'PopZ_Pattern_Scanning':fig1, 'Hotspot_plot':fig2,
        #             'Random_check':fig3, 'Tracker_plot':fig4, 'Unipolarity':fig0}
        #Pole_background(data_out['profile_folder'])
        figfolder = {'ProdPlot':fig1d}
        #Auto_HousingData(filename, figfolder, methodname, ctrl)
        ctrl -= 1
    
