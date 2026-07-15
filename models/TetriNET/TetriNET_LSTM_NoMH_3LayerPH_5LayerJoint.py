import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import io
import sys
import os
import keras
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from keras import layers
from numpy import array
from numpy.random import uniform
from numpy import hstack
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import tensorflow as tf
print(tf. __version__)
print(keras.__version__)
from tensorflow.keras.models import load_model

from pieceCoords import getPieceCoords
#Params###################################

LR_ARG = float(sys.argv[1])
BATCH_SIZE_ARG = int(sys.argv[2])
MAX_RUN_LEN_ARG = int(sys.argv[3])
EPOCHS_BASE_ARG = int(sys.argv[4])
EPOCHS_STEP_ARG = int(sys.argv[5])
EPOCHS_STEP_NUM_ARG = int(sys.argv[6])
NUM_GAMES_ARG = sys.argv[7]
MSG_WINDOW_LEN = int(sys.argv[8])
LAYER_NUM_ARG = int(sys.argv[9])
GPU_ARG = int(sys.argv[10])
inputfile = "TetriFeats250-289.txt"

DROPOUT=0
FINE_TUNE=False

LOAD_PREV_DATASET=True
#PREV_DATASET1="/home/users/abh61/tetrinet_train/dataset_dbg_16"
#PREV_DATASET1="/home/users/abh61/tetrinet_train/dataset_base10_holdout19_MsgCoordsLSTM_pad230_Window20"
PREV_DATASET1="/home/users/abh61/tetrinet_train/dataset_dense_ext_50-89_MsgCoords_20Window_LSTM_pad230_PieceFix"
PREV_DATASET2="/home/users/abh61/tetrinet_train/dataset_dense_ext_90-129_MsgCoords_20Window_LSTM_pad230_PieceFix"
PREV_DATASET3="/home/users/abh61/tetrinet_train/dataset_dense_ext_130-169_MsgCoords_20Window_LSTM_pad230_PieceFix"
PREV_DATASET4="/home/users/abh61/tetrinet_train/dataset_dense_ext_250-289_MsgCoords_20Window_LSTM_pad230_PieceFix"
#PREV_DATASET5="/home/users/abh61/tetrinet_train/dataset_ext_12_MsgCoords_3Window_LSTM_pad230_PieceFix"
#PREV_DATASET6="/home/users/abh61/tetrinet_train/dataset_ext_13_MsgCoords_3Window_LSTM_pad230_PieceFix"
#PREV_DATASET5="/home/users/abh61/tetrinet_train/dataset_ext_base10Minus14_MsgCoords_3Window_LSTM_pad230_PieceFix"
#PREV_DATASET5="/home/users/abh61/tetrinet_train/dataset_3FixedMH_base10V2_not14_MsgCoordsLSTM_pad230_MW20_PieceFix"
#PREV_DATASET5="/home/users/abh61/tetrinet_train/retrain_dataset_3FixedMH_base10V2_not16_MsgCoordsLSTM_pad230_MW20_PieceFix"

MSG_FEATS_SIZE=10 #Todo --make so this isn't redefined when LOAD_PREV_DATASET is false
numFeat=76 #Todo -- fix definition when LOAD_PREV_DATASET is false
MESSAGE_HISTORY_SIZE=32
JOINT_ACTIVATION="relu"
JOINT_SIZE=64
ctr = 0
for a in sys.argv:
    ctr = ctr + 1
    print("ARG " + str(ctr) + " : " + str(a))

model_base_name = './model_tetrinet_LSTM_' + 'COORDS_SCALED_EXT_' +str(LAYER_NUM_ARG) + '_Layer_BBBOW_MH_' + str(EPOCHS_BASE_ARG) + 'EP_LR_' + str(LR_ARG) + '_BATCH' + str(BATCH_SIZE_ARG) +     '_G' + str(NUM_GAMES_ARG) + '_PL' + str(MAX_RUN_LEN_ARG) + '_MW'+  str(MSG_WINDOW_LEN) + '_5LayerMH_'  +'_JOINTACTIVATION' + str(JOINT_ACTIVATION) + '_JOINTSIZE' + str(JOINT_SIZE) + '_DROPOUT' + str(DROPOUT)

if FINE_TUNE:
    model_base_name = model_base_name + '_FINETUNE'


print("Model base name is " + str(model_base_name))
print("input file is " + str(inputfile))
#print("MESSAGE_HISTORY_SIZE is " + str(MESSAGE_HISTORY_SIZE))
print("tensor_float_32: " + str( tf.config.experimental.tensor_float_32_execution_enabled()))

#############
#From https://www.tensorflow.org/guide/gpu

gpus = tf.config.list_physical_devices('GPU')
if gpus:

    try:
        enabled1 = tf.config.experimental.get_memory_growth(gpus[GPU_ARG])
        print("Enabled1: " + str(enabled1))
        tf.config.experimental.set_memory_growth(gpus[GPU_ARG], True)
        enabled2 = tf.config.experimental.get_memory_growth(gpus[GPU_ARG])

        #lim = 16 * 1024
        
        #tf.config.experimental.set_virtual_device_configuration(
         #   gpus[GPU_ARG], 
         #   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=lim)]
        #)
        
        print("Enabled2: " + str(enabled2))
        
        tf.config.set_visible_devices(gpus[GPU_ARG], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


if FINE_TUNE:
    prev_model_base_name = 'model_tetrinet_LSTM_3_Layer_BBBOW_MH_150EP_LR_0.001_BATCH64_G50-79_3LayerJointReluMsgFeatDrop.1_PL230_MW303LayerMH_MH_Size32_JOINTACTIVATIONrelu_JOINTSIZE32_DROPOUT0.1_'
    #prev_model_base_name = 'model_tetrinet_LSTM_3_Layer_BBBOW_MH_100EP_LR_0.001_BATCH64_G50-79_3LayerJointReluMsgFeat_PL230_MW303LayerMH_MH_Size32_JOINTACTIVATIONrelu_JOINTSIZE32_'
    print("Previous model name used as input for fine tuning: " + str(prev_model_base_name))
    suff = '_ckpt80.keras'
    prev_joint_model = load_model( prev_model_base_name + 'joint' + suff)
    prev_mh_model =    load_model( prev_model_base_name + 'mh'    + suff)
    prev_ph_model =    load_model( prev_model_base_name + 'ph'    + suff)

    ##########################################################################
    
    print("---Joint Model Details: " + str(prev_joint_model.summary()))
    print("Joint Model Weights: " +str(prev_joint_model.get_weights()))
    
    
    prev_dense_0_weights = prev_joint_model.get_layer('dense').get_weights()
    prev_dense_mid_weights = prev_joint_model.get_layer('dense_1').get_weights()
    prev_dense_mid2_weights= prev_joint_model.get_layer('dense_2').get_weights()
    prev_dense_1_weights = prev_joint_model.get_layer('dense_3').get_weights()
    print("Dense 0 saved weights: " + str(prev_dense_0_weights))
    print("Dense 1 saved weights: " + str(prev_dense_mid_weights))
    print("Dense 2 saved weights: " + str(prev_dense_mid2_weights))
    print("Dense 3 saved weights: " + str(prev_dense_1_weights))
    
    ##########################################################################
    print("---Message History Model Details: " + str(prev_mh_model.summary()))
    print("Message History Model Weights: " +str(prev_mh_model.get_weights()))
    prev_mh_lstm0_weights = prev_mh_model.get_layer('lstm').get_weights()
    prev_mh_lstm1_weights = prev_mh_model.get_layer('lstm_1').get_weights()
    print("LSTM 0 saved weights: " + str(prev_mh_lstm0_weights))
    print("LSTM 1 saved weights: " + str(prev_mh_lstm1_weights))

    ##########################################################################
    #Confusingly, the lstms in our PH oracle are named 2-4
    print("---Path History Model Details: " + str(prev_ph_model.summary()))
    print("Path History Model Weights: " +str(prev_ph_model.get_weights()))

    prev_ph_lstm0_weights = prev_ph_model.get_layer('lstm_2').get_weights()
    prev_ph_lstm1_weights = prev_ph_model.get_layer('lstm_3').get_weights()
    prev_ph_lstm2_weights = prev_ph_model.get_layer('lstm_4').get_weights()

    print("LSTM 2 saved weights: " + str(prev_ph_lstm0_weights))
    print("LSTM 3 saved weights: " + str(prev_ph_lstm1_weights))
    print("LSTM 4 saved weights: " + str(prev_ph_lstm2_weights))

    ##########################################################################

if LOAD_PREV_DATASET == False:        
    #Load the input data######################
    orig_data = pd.read_csv(inputfile, header=None)
    print("Loaded data")
    orig_data.columns = ['IsValidState', 'ID', 'ParentID', 'RandomSeedInit','MsgCtr', 'PrevBB','CurrBB', 'BranchDepth', 'BranchesSinceLastMsg', 'BranchType', 'LastIRBranchType', 'IsTerminal', 'RawMsg', 'XDim', 'YDim', 'Rotation', 'MsgType','Empty']
#HELPER FNs
IS_VALID_IDX = 0
ID_IDX = 1
PARENT_ID_IDX = 2
SEED_IDX = 3
MSG_CTR_IDX = 4
PREV_BB_IDX = 5
CURR_BB_IDX = 6
BRANCH_DEPTH_IDX = 7
BRANCHES_SINCE_LAST_MSG_IDX = 8
BRANCH_TYPE_IDX = 9
LAST_IR_BRANCH_IDX = 10
IS_TERMINAL_IDX = 11
RAW_MSG_IDX = 12
XDIM_IDX = 13
YDIM_IDX = 14
ROT_IDX = 15
PIECE_IDX = 16


#Take a name (string) and one-hot encoder ohe, and print a cpp map with the mappings
def genOHEEncoder(name, ohe):
    mapName = name + "EncodingMap"
    mapStr = "std::map<double, std::vector<float>> " + mapName + " = {\n"
    raw_vals = np.asarray(ohe.categories_).T
    for raw_val in raw_vals:
        lineStr =  "{" + str(raw_val[0]) + ", std::vector<float> {"
        enc_vals = ohe.transform(np.reshape(raw_val, (1,1)))
        #print(np.shape(enc_vals))
        enc_vals = enc_vals.flatten()
        vecCtr = 0
        for enc_val in enc_vals:
            vecCtr += 1
            lineStr = lineStr + str(enc_val)

            if (vecCtr != enc_vals.size):
                lineStr = lineStr +","
        lineStr = lineStr + "}"

        if (raw_val != raw_vals[-1]):
            lineStr = lineStr + "},\n"
        mapStr += lineStr

    mapStr += "}\n};"

    print(mapStr)

    #Todo -- handle failure if mapping not found
    code = "std::vector<float> oneHotEncode" + name + "(double val) { \n"
    code += " return " + mapName +".find(val)->second;" + "\n}"
    print(code)

#Construct the path/indicator lists given traces from past executions
def genPathsAndIndicators(inputs,gameList, maxRunLen):
    l_run_lens = []
    l_valid_run_lens = []

    #Shape is (run, timesteps, features) for paths
    l_paths = []
    #Shape is (run, timesteps) for indicators
    l_indicators = []

    #For padding
    zeroVec = [0.] * inputs.shape[1]
    LONGEST_RUN_LEN = np.max(inputs[:,BRANCHES_SINCE_LAST_MSG_IDX])
    print("Longest run is " + str(LONGEST_RUN_LEN))

    for currGameID in gameList:
        currGame = inputs[np.where(inputs[:, SEED_IDX] == currGameID)]

        print("Curr game idx is " + str(currGameID) + " with " + str(len(currGame)) + " rows")

        #Decide which message indexes to include based on the number of branches for that message
        l_picked_msg_idxs = []
        msg_idx_vals = np.unique(currGame[:,MSG_CTR_IDX])
        for i in (msg_idx_vals):
            curr_idxs = np.where(currGame[:, MSG_CTR_IDX] == i)
            curr_rows = currGame[curr_idxs]
            curr_valid_rows = curr_rows[np.where(curr_rows[:,0] ==1)]
            if (len(curr_valid_rows) < maxRunLen):
                if (len(curr_valid_rows) != 0):
                    l_picked_msg_idxs.append(i)
                    l_valid_run_lens.append(len(curr_valid_rows))
            if (len(curr_valid_rows) != 0):
                l_run_lens.append(len(curr_valid_rows))

        for currMsgIdx in l_picked_msg_idxs:
            currRows = currGame[np.where(currGame[:,MSG_CTR_IDX] == currMsgIdx)]
            #print("Looping at top with currMsgIdx " + str(currMsgIdx))
            #print("currRows is len " + str(len(currRows)))
            #Walk back from each record
            for i in range(0, currRows.shape [0] ):

                #print("Terminal branch at row " + str(i))
                curr_path = []
                curr_ind = []

                curr_ind.append(currRows[i][0])  #Valid or not
                curr_path.append(currRows[i].tolist())
                parentID = currRows[i][PARENT_ID_IDX]

                #Append parent branches
                for j in range (i -1, -1, -1):
                    if (currRows[j][ID_IDX] == parentID):
                        curr_path.append(currRows[j].tolist())
                        parentID = currRows[j][PARENT_ID_IDX]
                        curr_ind.append(currRows[j][0])

                #At this point we reverse and then pad with zeros
                curr_path.reverse()

                #Pad to length of longest path
                curr_path_len = len(curr_path)
                #print("Curr_path_len is " + str(curr_path_len))
                #for k in range (curr_path_len, LONGEST_RUN_LEN):
                for k in range (curr_path_len, maxRunLen):
                    curr_path.append(zeroVec)
                    curr_ind.append(0)

                l_paths.append(curr_path)
                l_indicators.append(curr_ind[0])

    return l_paths, l_indicators

def getTetrinetPieceVec(piece,pieceEncoder):
      piece_enc = pieceEncoder.transform(np.reshape(piece, (1,1)))
      piece_enc = piece_enc[0].astype('float32')
      #print(cmd_enc)
      #print(sampleRow[MSG_LEN_IDX])
      #res = np.concatenate((piece_enc, np.array([sampleRow[MSG_LEN_IDX]])))
      #res = np.concatenate(cmd_enc, sampleRow[MSG_LEN_IDX])
      #print("mavlinkMsgVec " + str(res))
      return piece_enc

# returns table of [Games,MsgCtr,PrevMsgIdxs,message_feats ]
def genPrevMsgWindowTable(runs,msgWindowSize,msgFeatsSize,pieceEncoder) :
    games = np.unique(runs[:,SEED_IDX])
    maxMessages = np.max(runs[:, MSG_CTR_IDX])
    
    prevMsgWindowTable = np.zeros((int(np.max(games)) +1,int(maxMessages) +1, msgWindowSize,msgFeatsSize))
    print("msgWindowTable has dims " + str(np.shape(prevMsgWindowTable)))
    
    for currGameID in range(0, int(np.max(games) +1)):
        print("currGameID is " + str(currGameID))
        currGame = runs[np.where((runs[:, SEED_IDX] == currGameID) & (runs[:,0] == 1))]
        if (len(currGame) == 0):
            print("No games for seed " + str(currGameID))
            continue
        for msgCtr in range(0, int(maxMessages) +1):
            print("msgCtr is " + str(msgCtr))
            if (msgCtr == 0 ):
                continue
            
            backMsgCtr = msgCtr
            #Walk backwards
            currWindow = np.zeros((msgWindowSize, msgFeatsSize))
            
            for i in range(0, msgWindowSize):
                backMsgCtr -= 1
                if (backMsgCtr < 0):
                    break
                
                #Go until we find a message
                while (backMsgCtr >=0):
                    prevRecords = currGame[np.where((currGame[:, MSG_CTR_IDX] == backMsgCtr))]
                    if (len(prevRecords) == 0):
                        backMsgCtr -= 1
                        continue
                    else:
                        
                        sampleRow = prevRecords[-1]
                        oneHotPiece = getTetrinetPieceVec(sampleRow[PIECE_IDX],pieceEncoder)
                        res = np.concatenate (([sampleRow[XDIM_IDX], sampleRow[YDIM_IDX], sampleRow[ROT_IDX]], oneHotPiece ), axis=0)
                        currWindow[i] = res
                        break
                      

            prevMsgWindowTable[currGameID,msgCtr] = np.flip(currWindow, axis=0)
                      

    return prevMsgWindowTable

def getPrevMsgWindow(game, msgCtr):
    return msgWindowTable[int(game), int(msgCtr)]

leftOffset = [
    [2,0,2,0],
    [0,0,0,0],
    [1,1,1,0],
    [1,1,1,0],
    [1,1,1,1],
    [1,1,1,0],
    [1,1,1,0]
]

rightOffset = [
    [1,0,1,0],
    [1,1,1,1],
    [1,0,1,1],
    [1,0,1,1],
    [1,0,1,0],
    [1,0,1,0],
    [1,0,1,1]
]

upOffset = [
    [0,0,0,0],
    [0,0,0,0],
    [0,1,1,1],
    [0,1,1,1],
    [0,1,0,1],
    [0,1,0,1],
    [0,1,1,1]
]

downOffset = [
    [0,3,0,3],
    [1,1,1,1],
    [1,1,0,1],
    [1,1,0,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,0,1]
]

#Given row, translate last 7 value back from one-hot vector to a single number
#e.g., [X,Y,Rot,0,1,0,0,0,0,0] -> 1
def windRowToPieceID(row):
    if (len(row) != 10):
        print("Error: Unexpected row length " + str(len(row)))
        quit()
    pieceEnc = row[3:]
    sum = np.sum(pieceEnc)
    if (sum != 1):
        print("Error: sum of 1-hot vec is " + str(sum))
        quit()
        
    return np.argmax(pieceEnc)

#return the coordinates of the leftmost, rightmost, and up/down coordinates for a piece
def getPrevMsgWindowEdges(x,y,rot, piece):
    if (not(piece == 0 or piece == 1 or piece == 2 or piece == 3 or piece == 4 or piece == 5 or piece == 6)):
        print("ERROR: Unknown piece " + str(piece))
        quit()
    if (not(rot == 0 or rot == 1 or rot == 2 or rot == 3)):
        print("ERROR: Unknown rotation " + str(rot))
        quit()
    xLeftEdge = x - leftOffset[piece][rot]
    xRightEdge = x + rightOffset[piece][rot]
    yUpEdge = y - upOffset[piece][rot]                  #top of tetrinet field is 0, bottom is 20 so subtract
    yDownEdge = y + downOffset[piece][rot]              
    return np.array([xLeftEdge, xRightEdge, yUpEdge, yDownEdge])

def getPrevMsgEdgesFromWindow(wind,maxWindSize):
    print("Window len is " + str(len(wind)))
    res = np.array([])
    for i in range(0, len(wind)):
        windX = wind[i][0]
        windY = wind[i][1]
        windRot = wind[i][2]
        if (windX == 0 and windY == 0 and windRot == 0):
            print("Zeros!")
            res = np.append(res,np.array([0,0,0,0]))
        else:
            windPiece = windRowToPieceID(wind[i]) 
            res = np.append(res,getPrevMsgWindowEdges(int(windX),int(windY),int(windRot),int(windPiece)))
        
    #Pad with zeros if the maxWindSize is bigger than current window
    #Does this actually trigger?
    for i in range(len(wind), maxWindSize):
        res = np.append(res, np.array([0,0,0,0]))
            
    return res

def getPrevMsgCoordsFromWindow(wind, maxWindSize):
    print("Window len is " + str(len(wind)))
    res = np.array([])
    for i in range(0, len(wind)):
        windX = wind[i][0]
        windY = wind[i][1]
        windRot = wind[i][2]
        if (windX == 0 and windY == 0 and windRot == 0):
            print("Zeros!")
            res = np.append(res,np.array([0,0,0,0,0,0,0,0]))
        else:
            windPiece = windRowToPieceID(wind[i])
            res = np.append(res,getPieceCoords(int(windX),int(windY),int(windRot),int(windPiece)))
    #Pad with zeros if the maxWindSize is bigger than current window
    #Does this actually trigger?
    for i in range(len(wind), maxWindSize):
        res = np.append(res, np.array([0,0,0,0,0,0,0,0]))
    return res
        

def getRotEnc(rot):
    if (not(rot == 0 or rot == 1 or rot == 2 or rot == 3)):
        print("Unexpected rotation " + str(rot) + " in getRotEnc")
        quit()
    if rot == 0:
        return np.array([1.0,0.0,0.0,0.0])
    elif rot == 1:
        return np.array([0.0,1.0,0.0,0.0])
    elif rot == 2:
        return np.array([0.0,0.0,1.0,0.0])
    elif rot == 3:
        return np.array([0.0,0.0,0.0,1.0])
    
def getPieceEnc(piece):
    if (not(piece == 0 or piece == 1 or piece == 2 or piece == 3 or piece == 4 or piece == 5 or piece == 6)):
        print("Unexpected piece " + str(piece) + " in getPieceEnc")
        quit()

    if piece == 0:
        return np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0])
    elif piece == 1:
        return np.array([0.0,1.0,0.0,0.0,0.0,0.0,0.0])
    elif piece == 2:
        return np.array([0.0,0.0,1.0,0.0,0.0,0.0,0.0])
    elif piece == 3:
        return np.array([0.0,0.0,0.0,1.0,0.0,0.0,0.0])
    elif piece == 4:
        return np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0])
    elif piece == 5:
        return np.array([0.0,0.0,0.0,0.0,0.0,1.0,0.0])
    elif piece == 6:
        return np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0])

###############################
#New stuff
if LOAD_PREV_DATASET == False:
    #MSG_WINDOW_SIZE = 20
    #############
    #Drop empty column at end of data
    #data_tmp = np.delete(orig_data,17,1) #empty column
    data_tmp = orig_data

    data_tmp.drop(data_tmp.columns[[17]], axis=1, inplace=True)

    #############
    base_feat_len = data_tmp.shape[1]
    piece_types = len(pd.unique(data_tmp['MsgType']))
    BB_types = len(pd.unique(data_tmp['CurrBB']))
    print(str(base_feat_len) + " base_feat_len")
    print(str(BB_types) + "BB types")
    #################
    #Create a 1-hot encoding for piece type
    #(MsgType is the ID of the piece)
    OHEPiece = OneHotEncoder(sparse_output = False)
    transformerPiece = make_column_transformer(
        (OHEPiece, ['MsgType']),
        #remainder='passthrough'
    )
    #Stick 1-hot piece encoding on end
    outPiece = pd.DataFrame(transformerPiece.fit_transform(data_tmp))
    concat1 = pd.concat([data_tmp,outPiece], axis="columns")
    concat1.columns = concat1.columns.astype(str)
    ################
    #Create 1-hot encoding for basic block
    OHECurrBB = OneHotEncoder(sparse_output = False)
    transformerCurrBB = make_column_transformer(
        (OHECurrBB, ['CurrBB']),
        #remainder='passthrough'
    )
    #Stick 1-hot basic block encoding on end
    outCurrBB = pd.DataFrame(transformerCurrBB.fit_transform(concat1))
    concat2 = pd.concat([concat1,outCurrBB], axis="columns") #Just get current BB
    final_data = concat2.to_numpy()
    ###############
    #Print our encoders
    mp = transformerCurrBB.named_transformers_
    BBEncoder = mp.get('onehotencoder')

    z = transformerPiece.named_transformers_
    pieceTransformer = z.get('onehotencoder')

    genOHEEncoder("Command",pieceTransformer )
    genOHEEncoder("CurrBB",BBEncoder )
    ##########################
    #Generate table with previous msg history
    piece_columns = final_data[:,PIECE_IDX]
    piece_columns = np.reshape(piece_columns, (len(piece_columns),1))
    #Size of msg feats is size of piece encoding (7) plus columns for x,y,z (3)
    print("Number of unique piece column values is " + str(np.unique(piece_columns[:,0])))
    print(np.unique(piece_columns[:,0]))
    MSG_FEATS_SIZE = len(np.unique(piece_columns[:,0])) + 3
    print("MSG_FEATS_SIZE is " + str(MSG_FEATS_SIZE))

    if (MSG_WINDOW_LEN != 0 ):
        #msgWindowTable = genPrevMsgWindowTable(final_data,MSG_WINDOW_LEN, MSG_FEATS_SIZE, pieceTransformer)
        msgWindowTable = genPrevMsgWindowTable(final_data,MSG_WINDOW_LEN, MSG_FEATS_SIZE, pieceTransformer)
        ###################
        #Zero out the raw msg string since we have its fields elsewhere
        #(and it messes up the numpy encoding of the data as floats)

    final_data[:,RAW_MSG_IDX] = 0
    ###########################

    #Removing 14 for now bc it has an overhang
    #Removing 14 for now bc it has an overhang
    #l_paths, l_indicators = genPathsAndIndicators(final_data, [19,20,21,22,23], MAX_RUN_LEN_ARG)
    #l_paths, l_indicators = genPathsAndIndicators(final_data, [50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89], MAX_RUN_LEN_ARG)
    #l_paths, l_indicators = genPathsAndIndicators(final_data, [90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129], MAX_RUN_LEN_ARG)
    #l_paths, l_indicators = genPathsAndIndicators(final_data, [130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169], MAX_RUN_LEN_ARG)
    #l_paths, l_indicators = genPathsAndIndicators(final_data, [15], MAX_RUN_LEN_ARG)
    #l_paths, l_indicators = genPathsAndIndicators(final_data, [12,13,14,15,19,20,21,22,23], MAX_RUN_LEN_ARG)
    l_paths, l_indicators = genPathsAndIndicators(final_data, [250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289], MAX_RUN_LEN_ARG)

    
    #l_paths, l_indicators = genPathsAndIndicators(final_data, [90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129], MAX_RUN_LEN_ARG)
    #l_paths, l_indicators = genPathsAndIndicators(final_data, [13,15,17,18,19,20,21,22,23], MAX_RUN_LEN_ARG)
    
    import gc
    pathArr = np.array(l_paths)
    indicArr = np.array(l_indicators)
    indicArr = np.reshape(indicArr,(np.shape(indicArr)[0],  1 ))
    
    print(np.shape(pathArr))
    print(np.shape(indicArr))
    import math
    split = math.floor(.8 * np.shape(pathArr) [0])
    print("Split is " + str(split))
    
    del l_paths
    del l_indicators
    gc.collect()

    floatIn = np.asarray(pathArr).astype('float32')
    floatOut = np.asarray(indicArr).astype('float32')

    #######################
    #Add in cumulative BB_BOW representation (sum of previous BB encodings as BOW)
    #Assumes post sequence padding
    bb_enc_off = base_feat_len + piece_types
    numFeat = np.shape(floatIn) [2]
    print("numfeat is " + str(numFeat))
    floatInExt = np.zeros((np.shape(floatIn)[0],np.shape(floatIn)[1], np.shape(floatIn)[2] +BB_types))
    #masking = layers.Masking(mask_value=0.0, input_shape=(MAX_RUN_LEN,numFeat))
    masking = layers.Masking(mask_value=0.0, input_shape=(np.shape(floatIn)[1],numFeat))

    last_bb_enc_train = np.zeros((np.shape(floatInExt)[0],BB_types))

    cum_bb_bow_train = np.zeros((np.shape(floatInExt)[0],BB_types))
    
    print("DBG1 ")
    for i in range(0, np.shape(floatIn)[0]):
        BB_BOW = np.zeros(BB_types) #Width is equal to number of items in BOW mapping
        #Figure out last nonmasked index
        masks = masking.compute_mask(floatIn[i])
        idx = np.argmin(masks)

        #We assume the 1 hot encoding  starts at bb_enc_off
        for j in range (0, idx):
            floatInExt[i,j] = np.append(floatIn[i,j], BB_BOW)
            cum_bb_bow_train[i] = BB_BOW #This is OK to get overwritten, will eventually hold cumulative bbbow for last timestep
            last_bb_enc_train[i] = floatIn[i,j,bb_enc_off:] #Should hold BB encoding for last timestep
            
            BB_BOW = BB_BOW + floatIn[i,j,bb_enc_off:]
            print("DBG2 ")
    ######################
    #Final prep for data sets prior to training

    #xtrain = floatInExt[:split]  WARNING REMOVE AFTER TESTING
    #ytrain = floatOut[:split]
    xtrain = floatInExt
    ytrain = floatOut
    xtest = floatInExt[split:]
    ytest = floatOut[split:]
    print("DBG3 ")
    ##############################################################


    #Get window of coords of pieces
    message_train = np.zeros((np.shape(xtrain)[0],MSG_WINDOW_LEN, 8))
    curr_message_train = np.zeros((np.shape(xtrain)[0],21))
    for i in range(0, np.shape(xtrain)[0]):
        currGame = xtrain[i][0][SEED_IDX]
        currMsgCtr = xtrain[i][0][MSG_CTR_IDX]
        currWind = getPrevMsgWindow(currGame, currMsgCtr)
        currWindCoordsFlat = getPrevMsgCoordsFromWindow(currWind, MSG_WINDOW_LEN) #This is a 1d array
        currWindCoords = currWindCoordsFlat.reshape((MSG_WINDOW_LEN,8))
        message_train[i] = currWindCoords
        
        curr_message_train_row = np.array([])
        curr_message_train_row = np.append(curr_message_train_row, xtrain[i][0][XDIM_IDX])
        curr_message_train_row = np.append(curr_message_train_row, xtrain[i][0][YDIM_IDX])
        curr_message_train_row = np.append(curr_message_train_row, getRotEnc(xtrain[i][0][ROT_IDX]))
        curr_message_train_row = np.append(curr_message_train_row, getPieceEnc(xtrain[i][0][PIECE_IDX]))
        curr_message_train_row = np.append(curr_message_train_row, getPieceCoords(xtrain[i][0][XDIM_IDX],xtrain[i][0][YDIM_IDX],xtrain[i][0][ROT_IDX],xtrain[i][0][PIECE_IDX]))
        curr_message_train[i] = curr_message_train_row
        if (i % 1000 == 0):
            print("For game " + str(currGame) + " msg " + str(currMsgCtr))
            print("message_train is " + str(message_train[i]))
            print("curr_message_train_row is " + str(curr_message_train[i]))
    
    #Zero some of the fields prior to training, as they're not available
    #at inference time
    xtrain[:,:,IS_VALID_IDX] = 0;
    xtrain[:,:,ID_IDX] = 0;
    xtrain[:,:,PARENT_ID_IDX] = 0;
    xtrain[:,:,SEED_IDX] = 0;
    xtrain[:,:,MSG_CTR_IDX] = 0;
    xtrain[:,:,PREV_BB_IDX] = 0;
    xtrain[:,:,CURR_BB_IDX] = 0;
    xtrain[:,:,BRANCH_DEPTH_IDX] = 0;
    
    xtrain[:,:,BRANCH_TYPE_IDX] = 0;
    xtrain[:,:,LAST_IR_BRANCH_IDX] = 0;
    xtrain[:,:,IS_TERMINAL_IDX] = 0;
    xtrain[:,:,PIECE_IDX] = 0;
    print("DBG5 ")
###########################################################
###########################################################
###########################################################
#Generate the architecture with the functional api

#2 or 3 Layer LSTM with message history

#print("DBG6 ")
#mh_masker = layers.Masking(mask_value=0.0, input_shape=(MSG_WINDOW_LEN, MSG_FEATS_SIZE))
#message_ins = Input(shape=(None,MSG_FEATS_SIZE), name="message_ins")
#mh_lstm0 = layers.LSTM(MESSAGE_HISTORY_SIZE, input_shape=(None,MSG_FEATS_SIZE), return_state=True, return_sequences=True)
#mh_lstm1 = layers.LSTM(MESSAGE_HISTORY_SIZE, input_shape=(None,MSG_FEATS_SIZE), return_state=True)

#mh_h0_in = Input(shape=(MESSAGE_HISTORY_SIZE), name= "mh_h0_in")
#mh_c0_in = Input(shape=(MESSAGE_HISTORY_SIZE), name= "mh_c0_in")
#mh_h1_in = Input(shape=(MESSAGE_HISTORY_SIZE), name= "mh_h1_in")
#mh_c1_in = Input(shape=(MESSAGE_HISTORY_SIZE), name= "mh_c1_in")

#m_message_ins = mh_masker(message_ins)
#mh_masked_val = mh_masker.compute_mask(message_ins)

#mh_tmp0_train, mh_h0_train, mh_c0_train = mh_lstm0(m_message_ins, mask=mh_masked_val)

#mh_tmp1_train, mh_h1_train, mh_c1_train = mh_lstm1(mh_tmp0_train)

#mh_tmp0_exp, mh_h0_exp, mh_c0_exp = mh_lstm0(m_message_ins, initial_state=[mh_h0_in, mh_c0_in], mask=mh_masked_val)
#mh_tmp1_exp, mh_h1_exp, mh_c1_exp = mh_lstm1(mh_tmp0_exp, initial_state=[mh_h1_in, mh_c1_in])

#Trick to make reading model outputs easier:
#mh_tmp1_exp_res = layers.Identity(name="mh_tmp1_exp_res")(mh_tmp1_exp)
#mh_h0_exp_res   = layers.Identity(name="mh_h0_exp_res") (mh_h0_exp)
#mh_c0_exp_res   = layers.Identity(name="mh_c0_exp_res") (mh_c0_exp)
#mh_h1_exp_res   = layers.Identity(name="mh_h1_exp_res") (mh_h1_exp)
#mh_c1_exp_res   = layers.Identity(name="mh_c1_exp_res") (mh_c1_exp)

#Only if fine tuning
#if FINE_TUNE:
#    mh_lstm0.set_weights(prev_mh_lstm0_weights)
#    mh_lstm1.set_weights(prev_mh_lstm1_weights)

    #Freeze message history representation for fine-tuning.
#    print('Freezing MH portion of model')
#    mh_lstm0.trainable=False
#    mh_lstm1.trainable=False

#model_mh_exp = Model(inputs=[message_ins, mh_h0_in, mh_c0_in, mh_h1_in, mh_c1_in], outputs=[mh_tmp1_exp_res, mh_h0_exp_res, mh_c0_exp_res, mh_h1_exp_res, mh_c1_exp_res])
#mh_model_exp = Model(inputs=[message_ins, mh_h0_in, mh_c0_in, mh_h1_in, mh_c1_in], outputs=[mh_tmp1_exp_res, mh_h0_exp_res, mh_c0_exp_res, mh_h1_exp_res, mh_c1_exp_res])



##################
print("Training Now .........................")

with tf.device("CPU"):
    #tens_dataset =tf.data.Dataset.from_tensor_slices(({"ph_feat_ins": xtrain,"message_ins": message_train, "curr_msg_ins": curr_message_train, "cum_bb_bow_ins": cum_bb_bow_train, "last_bb_enc_ins": last_bb_enc_train}, ytrain))

    
    #tens_dataset = tens_dataset.shuffle(buffer_size=100).batch(BATCH_SIZE_ARG)
    #tf.data.Dataset.save(tens_dataset, "/home/users/abh61/tetrinet_train/dataset_dense_ext_250-289_MsgCoords_20Window_LSTM_pad230_PieceFix",compression='GZIP')
    
    #quit()
    #tens_dataset.cache()
    #tens_dataset.prefetch(1)
    
    
    if LOAD_PREV_DATASET == True:
        tens_dataset1 = tf.data.Dataset.load(PREV_DATASET1,compression='GZIP')
        #tens_dataset = tens_dataset1
        tens_dataset2 = tf.data.Dataset.load(PREV_DATASET2,compression='GZIP')
        tens_dataset1_2 = tens_dataset1.concatenate(tens_dataset2)
        #tens_dataset = tens_dataset1_2
        tens_dataset3 = tf.data.Dataset.load(PREV_DATASET3,compression='GZIP')
        tens_dataset1_3 = tens_dataset1_2.concatenate(tens_dataset3)
        #tens_dataset = tens_dataset1_3
        tens_dataset4 = tf.data.Dataset.load(PREV_DATASET4,compression='GZIP')
        tens_dataset1_4 = tens_dataset1_3.concatenate(tens_dataset4)
        tens_dataset = tens_dataset1_4
        #tens_dataset5 = tf.data.Dataset.load(PREV_DATASET5,compression='GZIP')
        #tens_dataset1_5 = tens_dataset1_4.concatenate(tens_dataset5)
        #tens_dataset = tens_dataset1_5
        #tens_dataset6 = tf.data.Dataset.load(PREV_DATASET6,compression='GZIP')
        #tens_dataset1_6 = tens_dataset1_5.concatenate(tens_dataset6)
        #tens_dataset = tens_dataset1_6
        
    print(tens_dataset.element_spec)
                                                                                                                                
    ##############
    def get_stats(dataset, key):
        print(f"Calculating statistics for: {key}...")
        print("tmp1")
        spec = dataset.element_spec[0][key]
        num_features = spec.shape[-1]
        
        initial_min = tf.fill([num_features], tf.constant(np.inf, dtype=tf.float32))
        initial_max = tf.fill([num_features], tf.constant(-np.inf, dtype=tf.float32))
        print("tmp2")
        feature_ds = dataset.map(lambda x, y: tf.cast(x[key], tf.float32))
        print("tmp3")
        # Check if data is 2D (Time, Feat) or 3D (Batch, Time, Feat)
        # Most tf.data elements are 2D before batching
        def reduce_min_fn(state, x):
            # We use tf.range and tf.rank to stay purely in the TF domain
            # Or more simply: reduce over all axes except the last one
            dims = tf.range(tf.rank(x) - 1)
            sample_min = tf.reduce_min(x, axis=dims)
            return tf.minimum(state, sample_min)
        
        def reduce_max_fn(state, x):
            dims = tf.range(tf.rank(x) - 1)
            sample_max = tf.reduce_max(x, axis=dims)
            return tf.maximum(state, sample_max)
        print("tmp4")
        ds_min = feature_ds.reduce(initial_min, reduce_min_fn)
        print("tmp5")
        ds_max = feature_ds.reduce(initial_max, reduce_max_fn)
        print("tmp6")
        scale = 1.0 / (ds_max - ds_min + 1e-7)
        offset = -ds_min * scale
        print("tmp7")
        return scale, offset
    
    # Execute stats calculation for keys
    scale_ph, off_ph = get_stats(tens_dataset, "ph_feat_ins")
    scale_msg, off_msg = get_stats(tens_dataset, "message_ins")
    scale_curr, off_curr = get_stats(tens_dataset, "curr_msg_ins")
    scale_cum_bb_bow, off_cum_bb_bow = get_stats(tens_dataset, "cum_bb_bow_ins")
    scale_last_bb_enc, off_last_bb_enc = get_stats(tens_dataset, "last_bb_enc_ins")
    
    
    print(f"Unique scaling factors for ph_feat_ins: {scale_ph.shape}")  # Should be (76,)
    print(f"Unique scaling factors for message_ins: {scale_msg.shape}") # Should be (N,)
    print(f"Unique scaling factors for curr_msg_ins: {scale_curr.shape}") # Should be (N,)
    print(f"Unique scaling factors for cum_bb_bow: {scale_cum_bb_bow.shape}") # Should be (26,)
    print(f"Unique scaling factors for last_bb_enc: {scale_last_bb_enc.shape}") # Should be (26,)
    
    
    ##################
    
    
    tens_dataset = tens_dataset.shuffle(buffer_size=10000,seed=1)
    total_num = tens_dataset.cardinality().numpy()
    
    train_size = int(.9 * total_num)
    val_size = int(.1 * total_num)
    
    
    val_ds = tens_dataset.take(val_size)
    train_ds = tens_dataset.skip(val_size)
    #train_ds = remaining_ds.take(train_size)
    #train_ds = tens_dataset.take(train_size)
    #remaining_ds = tens_dataset.skip(train_size)
    #val_ds = remaining_ds.take(val_size)
    
    print("train_size is " + str(train_size) + " and val_size is " + str(val_size))
    print("train and val dataset sizes: " + str(train_ds.cardinality().numpy()) + "  " + str(val_ds.cardinality().numpy()))
    
    
    
    train_ds = train_ds.batch(BATCH_SIZE_ARG)
    val_ds = val_ds.batch(BATCH_SIZE_ARG) #Shouldn't affect the accuracy, needed to change shape
    print("Grabbed data")


##############################
MSG_FEATS_SIZE=8

print("DBG6 ")
mh_masker = layers.Masking(mask_value=0.0, input_shape=(MSG_WINDOW_LEN, MSG_FEATS_SIZE))
message_ins = Input(shape=(None,MSG_FEATS_SIZE), name="message_ins")
mh_lstm0 = layers.LSTM(MESSAGE_HISTORY_SIZE, input_shape=(None,MSG_FEATS_SIZE), return_state=True, return_sequences=True)
mh_lstm1 = layers.LSTM(MESSAGE_HISTORY_SIZE, input_shape=(None,MSG_FEATS_SIZE), return_state=True)

mh_h0_in = Input(shape=(MESSAGE_HISTORY_SIZE), name= "mh_h0_in")
mh_c0_in = Input(shape=(MESSAGE_HISTORY_SIZE), name= "mh_c0_in")
mh_h1_in = Input(shape=(MESSAGE_HISTORY_SIZE), name= "mh_h1_in")
mh_c1_in = Input(shape=(MESSAGE_HISTORY_SIZE), name= "mh_c1_in")

m_message_ins = mh_masker(message_ins)
mh_masked_val = mh_masker.compute_mask(message_ins)

#Scaling
mh_scaled = layers.Lambda(lambda val: val * scale_msg + off_msg, name=f"scale_message_ins")(m_message_ins)

mh_tmp0_train, mh_h0_train, mh_c0_train = mh_lstm0(mh_scaled, mask=mh_masked_val)
#mh_tmp0_train, mh_h0_train, mh_c0_train = mh_lstm0(m_message_ins, mask=mh_masked_val)

mh_tmp1_train, mh_h1_train, mh_c1_train = mh_lstm1(mh_tmp0_train)
mh_tmp0_exp, mh_h0_exp, mh_c0_exp = mh_lstm0(mh_scaled, initial_state=[mh_h0_in, mh_c0_in], mask=mh_masked_val)
#mh_tmp0_exp, mh_h0_exp, mh_c0_exp = mh_lstm0(m_message_ins, initial_state=[mh_h0_in, mh_c0_in], mask=mh_masked_val)
mh_tmp1_exp, mh_h1_exp, mh_c1_exp = mh_lstm1(mh_tmp0_exp, initial_state=[mh_h1_in, mh_c1_in])

#Trick to make reading model outputs easier:
mh_tmp1_exp_res = layers.Identity(name="mh_tmp1_exp_res")(mh_tmp1_exp)
mh_h0_exp_res   = layers.Identity(name="mh_h0_exp_res") (mh_h0_exp)
mh_c0_exp_res   = layers.Identity(name="mh_c0_exp_res") (mh_c0_exp)
mh_h1_exp_res   = layers.Identity(name="mh_h1_exp_res") (mh_h1_exp)
mh_c1_exp_res   = layers.Identity(name="mh_c1_exp_res") (mh_c1_exp)

model_mh_exp = Model(inputs=[message_ins, mh_h0_in, mh_c0_in, mh_h1_in, mh_c1_in], outputs=[mh_tmp1_exp_res, mh_h0_exp_res, mh_c0_exp_res, mh_h1_exp_res, mh_c1_exp_res])

###############################
if LOAD_PREV_DATASET == False:
    numFeat = np.shape(xtrain) [2]
ins = Input(shape=(None, numFeat), name="ph_feat_ins")

masking = layers.Masking(mask_value=0.0, input_shape=(MAX_RUN_LEN_ARG,numFeat))

lstm0 = layers.LSTM(32, input_shape=(None,numFeat),  return_state=True, return_sequences=True)
if LAYER_NUM_ARG == 2:
    lstm1 = layers.LSTM(32, input_shape=(None,numFeat),  return_state=True)
elif LAYER_NUM_ARG ==3:
    lstm1 = layers.LSTM(32, input_shape=(None,numFeat),  return_state=True,return_sequences=True)
    lstm2 = layers.LSTM(32, input_shape=(None,numFeat), return_state=True)

h0_in = Input(shape=(32), name="ph_h0_in")
c0_in = Input(shape=(32), name="ph_c0_in")
h1_in = Input(shape=(32), name="ph_h1_in")
c1_in = Input(shape=(32), name="ph_c1_in")
h2_in = Input(shape=(32), name="ph_h2_in")
c2_in = Input(shape=(32), name="ph_c2_in")
############################

#Need to be careful with mask and history vector; history could be populated even when branch feats are 0
m_ins = masking(ins)

masked_val = masking.compute_mask(ins)

#Scaling
scaled_m_ins = layers.Lambda(lambda val: val * scale_ph + off_ph, name=f"scale_ph_feat_ins")(m_ins)
tmp0_train, h0_train, c0_train = lstm0(scaled_m_ins, mask=masked_val)
#tmp0_train, h0_train, c0_train = lstm0(m_ins, mask=masked_val)
tmp1_train, h1_train, c1_train = lstm1(tmp0_train)
if LAYER_NUM_ARG == 3:
    tmp2_train, h2_train, c2_train = lstm2(tmp1_train)

    
#mh_vec_in = Input(shape=(MSG_WINDOW_LEN * MSG_FEATS_SIZE), name="message_ins")
##mh_vec_in = Input(shape=(MSG_WINDOW_LEN * 8), name="message_ins")
#fullX_exp = layers.concatenate([ins, mh_vec_in])
tmp0_exp, h0_out, c0_out = lstm0(scaled_m_ins, mask=masked_val, initial_state=[h0_in, c0_in])
#tmp0_exp, h0_out, c0_out = lstm0(m_ins, mask=masked_val, initial_state=[h0_in, c0_in])
tmp1_exp, h1_out, c1_out = lstm1(tmp0_exp, initial_state=[h1_in, c1_in])
if LAYER_NUM_ARG == 3:
    tmp2_exp, h2_out, c2_out = lstm2(tmp1_exp, initial_state=[h2_in,c2_in])
    

if LAYER_NUM_ARG == 2:
    tmp1_exp_res = layers.Identity(name="tmp1_exp_res") (tmp1_exp)
elif LAYER_NUM_ARG ==3:
    tmp2_exp_res = layers.Identity(name="tmp2_exp_res") (tmp2_exp)
h0_out_res = layers.Identity(name="h0_out_res") (h0_out)
c0_out_res = layers.Identity(name="c0_out_res") (c0_out)
h1_out_res = layers.Identity(name="h1_out_res") (h1_out)
c1_out_res = layers.Identity(name="c1_out_res") (c1_out)
if LAYER_NUM_ARG == 3:
    h2_out_res = layers.Identity(name="h2_out_res") (h2_out)
    c2_out_res = layers.Identity(name="c2_out_res") (c2_out)
curr_msg_ins = Input(shape=(21), name="curr_msg_ins")
cum_bb_bow_ins = Input(shape=(26), name="cum_bb_bow_ins")
last_bb_enc_ins = Input(shape=(26), name="last_bb_enc_ins")

#if LAYER_NUM_ARG == 2:
#    finalX = layers.concatenate([tmp1_train, mh_tmp1_train, curr_msg_ins])
#elif LAYER_NUM_ARG == 3:
#    finalX = layers.concatenate([tmp2_train, mh_tmp1_train, curr_msg_ins])

#NEW
#Scaling
scaled_cum_bb_bow_ins = layers.Lambda(lambda val: val * scale_cum_bb_bow + off_cum_bb_bow, name=f"scale_cum_bb_bow_ins")(cum_bb_bow_ins)
##scaled_mh_vec_in = layers.Lambda(lambda val: val * scale_msg + off_msg, name=f"scale_mh_vec_ins")(mh_vec_in)
scaled_curr_msg_ins = layers.Lambda(lambda val: val * scale_curr + off_curr, name=f"scale_curr_msg_ins")(curr_msg_ins)
scaled_last_bb_enc_ins = layers.Lambda(lambda val: val * scale_last_bb_enc + off_last_bb_enc, name=f"scale_last_bb_enc_ins")(last_bb_enc_ins)

if LAYER_NUM_ARG == 2:
    finalX = layers.concatenate([tmp1_train, mh_tmp1_train, scaled_curr_msg_ins, scaled_cum_bb_bow_ins, scaled_last_bb_enc_ins])
elif LAYER_NUM_ARG == 3:
    finalX = layers.concatenate([tmp2_train, mh_tmp1_train, scaled_curr_msg_ins, scaled_cum_bb_bow_ins, scaled_last_bb_enc_ins])
#if LAYER_NUM_ARG == 2:
#    finalX = layers.concatenate([tmp1_train, mh_vec_in, curr_msg_ins, cum_bb_bow_ins, last_bb_enc_ins])
#elif LAYER_NUM_ARG == 3:
#    finalX = layers.concatenate([tmp2_train, mh_vec_in, curr_msg_ins, cum_bb_bow_ins, last_bb_enc_ins])


        
d_final_0 = layers.Dense(JOINT_SIZE, activation=JOINT_ACTIVATION)
d_final_mid = layers.Dense(JOINT_SIZE,activation=JOINT_ACTIVATION)
d_final_mid2 = layers.Dense(JOINT_SIZE,activation=JOINT_ACTIVATION)
d_final_mid3 = layers.Dense(JOINT_SIZE,activation=JOINT_ACTIVATION)
d_final_mid4 = layers.Dense(JOINT_SIZE,activation=JOINT_ACTIVATION)
d_final_1 = layers.Dense(1, activation="sigmoid")

if DROPOUT != 0:
    #drop0 = layers.Dropout(DROPOUT)
    drop1 = layers.Dropout(DROPOUT)
    #drop2 = layers.Dropout(DROPOUT)
    final_outs = d_final_1(drop1(d_final_mid4(d_final_mid3(d_final_mid2(d_final_mid(d_final_0(finalX)))))))
    #final_outs = d_final_1(drop1(d_final_mid3(d_final_mid2(d_final_mid(d_final_0(finalX))))))
    #final_outs = d_final_1(drop1(d_final_mid2(d_final_mid(d_final_0(finalX)))))
else:
    #final_outs = d_final_1(d_final_mid2(d_final_mid(d_final_0(finalX))))
    #final_outs = d_final_1(d_final_mid3(d_final_mid2(d_final_mid(d_final_0(finalX)))))
    final_outs = d_final_1(d_final_mid4(d_final_mid3(d_final_mid2(d_final_mid(d_final_0(finalX))))))

    
path_summary    = Input(shape=(32), name="path_summary")
##message_summary = Input(shape=(MSG_WINDOW_LEN * 8), name="message_summary")
##message_summary_scaled = layers.Lambda(lambda val: val * scale_msg + off_msg, name=f"scale_message_summary")(message_summary)
message_summary = Input(shape=(MESSAGE_HISTORY_SIZE), name="message_summary")
current_message = Input(shape=(21), name="current_message")
current_message_scaled = layers.Lambda(lambda val: val * scale_curr + off_curr, name=f"scale_current_message")(current_message)

cum_bb_bow = Input(shape=(26), name="cum_bb_bow")
cum_bb_bow_scaled = layers.Lambda(lambda val: val * scale_cum_bb_bow + off_cum_bb_bow, name=f"cum_bb_bow_scaled")(cum_bb_bow)
last_bb = Input(shape=(26), name="last_bb")
last_bb_scaled = layers.Lambda(lambda val: val * scale_last_bb_enc + off_last_bb_enc, name=f"last_bb_scale")(last_bb)


#finalXExp = layers.concatenate([path_summary, message_summary, current_message])
#finalXExp = layers.concatenate([path_summary, message_summary, current_message, cum_bb_bow, last_bb])
##finalXExp = layers.concatenate([path_summary, message_summary_scaled, current_message_scaled, cum_bb_bow_scaled, last_bb_scaled])
finalXExp = layers.concatenate([path_summary, message_summary, current_message_scaled, cum_bb_bow_scaled, last_bb_scaled])

final_exp_out = d_final_1(d_final_mid4(d_final_mid3(d_final_mid2(d_final_mid(d_final_0(finalXExp))))))
#final_exp_out = d_final_1(d_final_mid3(d_final_mid2(d_final_mid(d_final_0(finalXExp)))))
#final_exp_out = d_final_1(d_final_mid2(d_final_mid(d_final_0(finalXExp))))

##model_train= Model(inputs=[ins, mh_vec_in,curr_msg_ins,cum_bb_bow_ins,last_bb_enc_ins], outputs=final_outs)
model_train= Model(inputs=[ins, message_ins,curr_msg_ins,cum_bb_bow_ins,last_bb_enc_ins], outputs=final_outs)
my_opt = tf.keras.optimizers.Adam(learning_rate=LR_ARG)
model_train.compile(loss="binary_crossentropy", optimizer=my_opt, metrics=['accuracy',tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]) #Set to crossentropy for classification

#model_all_exp  = Model(inputs[ins, h0_in, c0_in, h1_in, c1_in, h2_in, c2_in, message_ins, ])

model_train.summary()
if LAYER_NUM_ARG == 2:
    model_path_exp = Model(inputs=[ins,h0_in, c0_in, h1_in, c1_in], outputs=[ tmp1_exp_res, h0_out_res, c0_out_res,  h1_out_res,  c1_out_res])
elif LAYER_NUM_ARG == 3:
    model_path_exp = Model(inputs=[ins,h0_in, c0_in, h1_in, c1_in, h2_in, c2_in], outputs=[ tmp2_exp_res, h0_out_res, c0_out_res,  h1_out_res,  c1_out_res, h2_out_res,  c2_out_res])
    
model_joint_exp = Model(inputs=[path_summary, message_summary,current_message, cum_bb_bow, last_bb], outputs=[final_exp_out])
model_path_exp.summary()
model_joint_exp.summary()

    
    
#model_train.fit(xtrain, ytrain, epochs=2,  verbose=1)
#model_exp.save('/home/users/abh61/model_LSTM_3Layer_BBBOW_2EP_WithPiece')
class EpochSnapshotCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs=None):
        print("Calling callback at end of epoch " + str(epochs))
        if (epochs % EPOCHS_STEP_ARG == 0):
            #print("Not Saving models")
            #model_exp.save(model_base_name + '_ckpt' + str(epochs) + 'no_history')
            model_mh_exp.save(model_base_name + '_mh_ckpt' + str(epochs))
            model_path_exp.save(model_base_name + '_ph_ckpt' + str(epochs))
            model_joint_exp.save(model_base_name + '_joint_ckpt' +str(epochs))
            model_mh_exp.save(model_base_name + '_mh_ckpt' + str(epochs) + '.keras')
            model_path_exp.save(model_base_name + '_ph_ckpt' + str(epochs) + '.keras')
            model_joint_exp.save(model_base_name + '_joint_ckpt' +str(epochs) + '.keras')
            print("Performance on validation set...")
            model_train.evaluate(val_ds)
                
                
#model_train.fit({"ph_feat_ins": xtrain,"message_ins": message_train}, ytrain, batch_size=256, epochs=EPOCHS_BASE_ARG,  verbose=1)                
#model_train.fit({"ph_feat_ins": xtrain,"message_ins": message_train, "curr_msg_ins": curr_message_train}, ytrain, epochs=EPOCHS_BASE_ARG, batch_size=BATCH_SIZE_ARG,  verbose=2, callbacks=[EpochSnapshotCallback()])
model_train.fit(train_ds, epochs=EPOCHS_BASE_ARG,  verbose=2, callbacks=[EpochSnapshotCallback()])

model_mh_exp.save(model_base_name + '_mh')
model_path_exp.save(model_base_name + '_ph' )
model_joint_exp.save(model_base_name + '_joint')
model_mh_exp.save(model_base_name + '_mh' + '.keras')
model_path_exp.save(model_base_name + '_ph'  + '.keras')
model_joint_exp.save(model_base_name + '_joint' + '.keras')

#model_exp.save(model_base_name + 'no_history')

#model_train.evaluate(xtest, ytest)
                
                



