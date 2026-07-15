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

#Params###################################

LR_ARG = float(sys.argv[1])
BATCH_SIZE_ARG = int(sys.argv[2])
MAX_PATHLEN_ARG = int(sys.argv[3])
EPOCHS_BASE_ARG = int(sys.argv[4])
EPOCHS_STEP_ARG = int(sys.argv[5])
EPOCHS_STEP_NUM_ARG = int(sys.argv[6])
NUM_GAMES_ARG = int(sys.argv[7])
MSG_WINDOW_LEN = int(sys.argv[8])
GPU_ARG = int(sys.argv[9])
inputfile = "pflmavlink.txt"

#############
#From https://www.tensorflow.org/guide/gpu

gpus = tf.config.list_physical_devices('GPU')
if gpus:

    try:
        tf.config.set_visible_devices(gpus[GPU_ARG], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


#Load the input data######################
orig_data = pd.read_csv(inputfile, header=None)
print("Loaded data")
orig_data.columns = ['IsValidState', 'ID', 'ParentID', 'Seed','MsgCtr', 'PrevBB','CurrBB', 'BranchDepth', 'BranchesSinceLastMsg', 'BranchType', 'LastIRBranchType', 'IsTerminal', 'MsgType', 'MsgLen']
#HELPER FNs
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
CMD_IDX = 12
MSG_LEN_IDX = 13



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
    print("Zeroing last branch type and raw msg")
    inputs[:,BRANCH_TYPE_IDX] = 0
    inputs[:, CMD_IDX] = 0
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

def getMavlinkMsgVec(sampleRow,msgEncoder):
    #print("entering getMavlinkMsgVec")
    cmd = sampleRow[CMD_IDX]
    cmd_enc = msgEncoder.transform(np.reshape(cmd, (1,1)))
    cmd_enc = cmd_enc[0].astype('float32')
    #print(cmd_enc)
    #print(sampleRow[MSG_LEN_IDX])
    res = np.concatenate((cmd_enc, np.array([sampleRow[MSG_LEN_IDX]])))
    #res = np.concatenate(cmd_enc, sampleRow[MSG_LEN_IDX])
    #print("mavlinkMsgVec " + str(res))
    return res

# returns table of [Games,MsgCtr,PrevMsgIdxs,message_feats ]
def genPrevMsgWindowTable(runs, msgWindowSize, msgFeatsSize, msgEncoder):
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
            #print("msgCtr is " + str(msgCtr))
            if (msgCtr == 0 ):
                continue
              
            backMsgCtr = msgCtr
            #Walk backwards
            #print("DBG 1")
            currWindow = np.zeros((msgWindowSize, msgFeatsSize))
            #print("DBG 2")
            for i in range(0, msgWindowSize):
                backMsgCtr -= 1
                #print("DBG 3")
                if (backMsgCtr < 0):
                    break

                #Go until we find a message
                while (backMsgCtr >=0):
                    #print("DBG 4")
                    prevRecords = currGame[np.where((currGame[:, MSG_CTR_IDX] == backMsgCtr))]
                    if (len(prevRecords) == 0):
                        #print("DBG 5")
                        backMsgCtr -= 1
                        continue
                    else:
                        #print("DBG 6")
                        sampleRow = prevRecords[-1]
                        #print("about to call getMavlinkMsgVec")
                        res = getMavlinkMsgVec(sampleRow, msgEncoder)
                        #oneHotPiece = getOneHotPieceEncoding(sampleRow[PIECE_COLUMN])
                        #res = np.concatenate (([sampleRow[XDIM_IDX], sampleRow[YDIM_IDX], sampleRow[ROT_IDX]], oneHotPiece ), axis=0)
                        currWindow[i] = res
                        break

            #prevMsgWindowTable[currGameID,msgCtr] = currWindow
            #Tabbing wrong?
            prevMsgWindowTable[currGameID,msgCtr] = np.flip(currWindow, axis=0)


    return prevMsgWindowTable

def getPrevMsgWindow(game, msgCtr):
    return msgWindowTable[int(game), int(msgCtr)]
###############################


base_feat_len = orig_data.shape[1]
command_types = len(pd.unique(orig_data['MsgType']))
BB_types = len(pd.unique(orig_data['CurrBB']))
print(str(base_feat_len) + " base_feat_len")
print(str(command_types) + " command types")
print(str(BB_types) + "BB types")

data_tmp = orig_data
#---------------------------------------
#Create a 1-hot encoding for message type
OHECommand = OneHotEncoder(sparse_output = False)
transformerCommand = make_column_transformer(
    (OHECommand, ['MsgType']),
    #remainder='passthrough'
)
#Stick 1-hot message type encoding on end
outCommand = pd.DataFrame(transformerCommand.fit_transform(data_tmp))
concat1 = pd.concat([data_tmp,outCommand], axis="columns")
concat1.columns = concat1.columns.astype(str)
#-----------------------------------------

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
#------------------------------------------

command_columns = final_data[:,12]
command_columns = np.reshape(command_columns, (len(command_columns), 1))
print("Commands are " + str(np.unique(command_columns[:,0])))
MSG_FEATS_SIZE = len(np.unique(command_columns[:,0])) + 1
mp = transformerCurrBB.named_transformers_
BBEncoder = mp.get('onehotencoder')

z = transformerCommand.named_transformers_
commandTransformer = z.get('onehotencoder')

genOHEEncoder("Command",commandTransformer )
genOHEEncoder("CurrBB",BBEncoder )

#-----------------
if (MSG_WINDOW_LEN != 0 ):
    msgWindowTable = genPrevMsgWindowTable(final_data,MSG_WINDOW_LEN, MSG_FEATS_SIZE, commandTransformer)
#---------------------------------------------------

l_paths, l_indicators = genPathsAndIndicators(final_data, [23,24,25,26,27,28,29,30], MAX_PATHLEN_ARG)

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
#------------------------------------
floatIn = np.asarray(pathArr).astype('float32')
floatOut = np.asarray(indicArr).astype('float32')

floatIn[:,:,0] = 0;
floatIn[:,:,ID_IDX] = 0;
floatIn[:,:,PARENT_ID_IDX] = 0;
floatIn[:,:,SEED_IDX] = 0;
floatIn[:,:,MSG_CTR_IDX] = 0;
floatIn[:,:,PREV_BB_IDX] = 0;
floatIn[:,:,CURR_BB_IDX] = 0;
floatIn[:,:,BRANCH_DEPTH_IDX] = 0;

floatIn[:,:,BRANCH_TYPE_IDX] = 0;
floatIn[:,:,LAST_IR_BRANCH_IDX] = 0;
floatIn[:,:,IS_TERMINAL_IDX] = 0;
floatIn[:,:,CMD_IDX] = 0;

#-----------------------------------------
#DOES NOT WORK WITH ZERO PADDING: FIX
#SANITY CHECK ME
#Assumes post sequence padding
bb_enc_off = base_feat_len + command_types
numFeat = np.shape(floatIn) [2]
print("numfeat is " + str(numFeat))
floatInExt = np.zeros((np.shape(floatIn)[0],np.shape(floatIn)[1], np.shape(floatIn)[2] +BB_types))
#masking = layers.Masking(mask_value=0.0, input_shape=(MAX_RUN_LEN,numFeat))
masking = layers.Masking(mask_value=0.0, input_shape=(np.shape(floatIn)[1],numFeat))
for i in range(0, np.shape(floatIn)[0]):
    BB_BOW = np.zeros(BB_types) #Width is equal to number of items in BOW mapping
    #Figure out last nonmasked index
    masks = masking.compute_mask(floatIn[i])
    idx = np.argmin(masks)

    #We assume the 1 hot encoding  starts at bb_enc_off
    for j in range (0, idx):
        floatInExt[i,j] = np.append(floatIn[i,j], BB_BOW)
        BB_BOW = BB_BOW + floatIn[i,j,bb_enc_off:]
        
        #xtrain = floatInExt[:split]  WARNING REMOVE AFTER TESTING
        #ytrain = floatOut[:split]
xtrain = floatInExt
ytrain = floatOut
xtest = floatInExt[split:]
ytest = floatOut[split:]

message_train = np.zeros((np.shape(xtrain)[0],MSG_WINDOW_LEN,MSG_FEATS_SIZE))
for i in range(0, np.shape(xtrain)[0]):
    #Get game and msgctr from any of the timestamps in batch i
    currGame = xtrain[i][0][SEED_IDX]
    currMsgCtr = xtrain[i][0][MSG_CTR_IDX]
    message_train[i] = getPrevMsgWindow(currGame, currMsgCtr)

#########################
#Zero some of the fields prior to training, as they're not available
#at inference time
    
xtrain[:,:,0] = 0;
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
xtrain[:,:,CMD_IDX] = 0;
    
xtest[:,:,0] = 0;
xtest[:,:,ID_IDX] = 0;
xtest[:,:,PARENT_ID_IDX] = 0;
xtest[:,:,SEED_IDX] = 0;
xtest[:,:,MSG_CTR_IDX] = 0;
xtest[:,:,PREV_BB_IDX] = 0;
xtest[:,:,CURR_BB_IDX] = 0;
xtest[:,:,BRANCH_DEPTH_IDX] = 0;

xtest[:,:,BRANCH_TYPE_IDX] = 0;
xtest[:,:,LAST_IR_BRANCH_IDX] = 0;
xtest[:,:,IS_TERMINAL_IDX] = 0;
xtest[:,:,CMD_IDX] = 0;


###################################################
#strategy = tf.distribute.MirroredStrategy()

#---------MESSAGE HISTORY

history_vec_size = 32


mh_masker = layers.Masking(mask_value=0.0, input_shape=(MSG_WINDOW_LEN, MSG_FEATS_SIZE))
message_ins = Input(shape=(None,MSG_FEATS_SIZE), name="message_ins")
mh_lstm0 = layers.LSTM(32, input_shape=(None,MSG_FEATS_SIZE), return_state=True, return_sequences=True)
mh_lstm1 = layers.LSTM(32, input_shape=(None,MSG_FEATS_SIZE), return_state=True)

mh_h0_in = Input(shape=32, name= "mh_h0_in")
mh_c0_in = Input(shape=32, name= "mh_c0_in")
mh_h1_in = Input(shape=32, name= "mh_h1_in")
mh_c1_in = Input(shape=32, name= "mh_c1_in")

m_message_ins = mh_masker(message_ins)
mh_masked_val = mh_masker.compute_mask(message_ins)

mh_tmp0_train, mh_h0_train, mh_c0_train = mh_lstm0(m_message_ins, mask=mh_masked_val)

mh_tmp1_train, mh_h1_train, mh_c1_train = mh_lstm1(mh_tmp0_train)

mh_tmp0_exp, mh_h0_exp, mh_c0_exp = mh_lstm0(m_message_ins, initial_state=[mh_h0_in, mh_c0_in], mask=mh_masked_val)
mh_tmp1_exp, mh_h1_exp, mh_c1_exp = mh_lstm1(mh_tmp0_exp, initial_state=[mh_h1_in, mh_c1_in])

#Trick to make reading model outputs easier:
mh_tmp1_exp_res = layers.Identity(name="mh_tmp1_exp_res")(mh_tmp1_exp)
mh_h0_exp_res   = layers.Identity(name="mh_h0_exp_res") (mh_h0_exp)
mh_c0_exp_res   = layers.Identity(name="mh_c0_exp_res") (mh_c0_exp)
mh_h1_exp_res   = layers.Identity(name="mh_h1_exp_res") (mh_h1_exp)
mh_c1_exp_res   = layers.Identity(name="mh_c1_exp_res") (mh_c1_exp)

model_mh_exp = Model(inputs=[message_ins, mh_h0_in, mh_c0_in, mh_h1_in, mh_c1_in], outputs=[mh_tmp1_exp_res, mh_h0_exp_res, mh_c0_exp_res, mh_h1_exp_res, mh_c1_exp_res])
##############################

numFeat = np.shape(xtrain) [2]
ins = Input(shape=(None, numFeat), name="ph_feat_ins")

masking = layers.Masking(mask_value=0.0, input_shape=(MAX_PATHLEN_ARG,numFeat))

lstm0 = layers.LSTM(32, input_shape=(None,numFeat),  return_state=True, return_sequences=True)
lstm1 = layers.LSTM(32, input_shape=(None,numFeat),  return_state=True)


h0_in = Input(shape=32, name="ph_h0_in")
c0_in = Input(shape=32, name="ph_c0_in")
h1_in = Input(shape=32, name="ph_h1_in")
c1_in = Input(shape=32, name="ph_c1_in")

############################

#Need to be careful with mask and history vector; history could be populated even when branch feats are 0
m_ins = masking(ins)

masked_val = masking.compute_mask(ins)
tmp0_train, h0_train, c0_train = lstm0(m_ins, mask=masked_val)
tmp1_train, h1_train, c1_train = lstm1(tmp0_train)



mh_vec_in = Input(shape=history_vec_size)
#fullX_exp = layers.concatenate([ins, mh_vec_in])
tmp0_exp, h0_out, c0_out = lstm0(m_ins, mask=masked_val, initial_state=[h0_in, c0_in])
tmp1_exp, h1_out, c1_out = lstm1(tmp0_exp, initial_state=[h1_in, c1_in])


tmp1_exp_res = layers.Identity(name="tmp2_exp_res") (tmp1_exp)
h0_out_res = layers.Identity(name="h0_out_res") (h0_out)
c0_out_res = layers.Identity(name="c0_out_res") (c0_out)
h1_out_res = layers.Identity(name="h1_out_res") (h1_out)
c1_out_res = layers.Identity(name="c1_out_res") (c1_out)

finalX = layers.concatenate([tmp1_train, mh_tmp1_train])

d_final_0 = layers.Dense(32)
d_final_1 = layers.Dense(1, activation="sigmoid")

final_outs = d_final_1(d_final_0(finalX))


path_summary    = Input(shape=32, name="path_summary")
message_summary = Input(shape=32, name="message_summary")
finalXExp = layers.concatenate([path_summary, message_summary])
final_exp_out = d_final_1(d_final_0(finalXExp))

model_train= Model(inputs=[ins, message_ins], outputs=final_outs)
model_train.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy',tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]) #Set to crossentropy for classification

#model_all_exp  = Model(inputs[ins, h0_in, c0_in, h1_in, c1_in, h2_in, c2_in, message_ins, ])

model_train.summary()
model_path_exp = Model(inputs=[ins,h0_in, c0_in, h1_in, c1_in], outputs=[ tmp1_exp_res, h0_out_res, c0_out_res,  h1_out_res,  c1_out_res])
#model_path_exp3 = Model(inputs=[ins,h0_in, c0_in, h1_in, c1_in, h2_in, c2_in], outputs={'tmp2_exp' :tmp2_exp,'h0_out' :h0_out,'c0_out': c0_out, 'h1_out': h1_out, 'c1_out' : c1_out, 'h2_out' : h2_out, 'c2_out': c2_out})
#model_path_exp2 = Model(inputs=[ins,h0_in, c0_in, h1_in, c1_in, h2_in, c2_in], outputs=[tmp2_exp, h2_out, c2_out])
model_joint_exp = Model(inputs=[path_summary, message_summary], outputs=[final_exp_out])
model_path_exp.summary()
model_joint_exp.summary()

#with strategy.scope():
# 2 Layer masked, no message history
#######################################################
print("Training Now .........................")
model_base_name = '/home/users/abh61/model_mavlink_LSTM_2Layer_BBBOW_MH_' + str(EPOCHS_BASE_ARG) + 'EP_LR_' + str(LR_ARG) + '_BATCH' + str(BATCH_SIZE_ARG) +     '_G' + str(NUM_GAMES_ARG) + '_PL' + str(MAX_PATHLEN_ARG) + '_MWFixed' + str(MSG_WINDOW_LEN)
#model_train.fit(xtrain, ytrain, epochs=2,  verbose=1)
#model_exp.save('/home/users/abh61/model_LSTM_3Layer_BBBOW_2EP_WithPiece')
class EpochSnapshotCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs=None):
        print("Calling callback at end of epoch " + str(epochs))
        if (epochs % EPOCHS_STEP_ARG == 0):
            print("Not saving models")
            #model_exp.save(model_base_name + '_ckpt' + str(epochs) + 'no_history')
            #model_mh_exp.save(model_base_name + '_mh_ckpt' + str(epochs))
            #model_path_exp.save(model_base_name + '_ph_ckpt' + str(epochs))
            #model_joint_exp.save(model_base_name + '_joint_ckpt' +str(epochs))
                
                
                
                
                
                
                
                
#model_train.fit({"ph_feat_ins": xtrain,"message_ins": message_train}, ytrain, batch_size=256, epochs=EPOCHS_BASE_ARG,  verbose=1)                
model_train.fit({"ph_feat_ins": xtrain,"message_ins": message_train}, ytrain, epochs=EPOCHS_BASE_ARG, batch_size=BATCH_SIZE_ARG,  verbose=2, callbacks=[EpochSnapshotCallback()])
                
model_mh_exp.save(model_base_name + '_mh')
model_path_exp.save(model_base_name + '_ph')
model_joint_exp.save(model_base_name + '_joint')

#model_exp.save(model_base_name + 'no_history')

#model_train.evaluate(xtest, ytest)
                
                



