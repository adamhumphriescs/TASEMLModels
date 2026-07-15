import numpy as np

def getPieceCoords(x,y,rot,piece):
    if (not(rot == 0 or rot == 1 or rot == 2 or rot == 3)):
        print("Unexpected rotation " + str(rot) + " in getPieceCoords")
        quit()
    if (not(piece == 0 or piece == 1 or piece == 2 or piece == 3 or piece == 4 or piece == 5 or piece == 6)):
        print("Unexpected piece " + str(piece) + " in getPieceCoords")
        quit()

    if (piece == 0):
        if (rot == 0):
            #  ##X#
            return np.array([x-2,y, x-1, y, x,y, x+1,y])
        elif (rot == 1):
            #  X
            #  #
            #  #
            #  #
            return np.array([x,y, x, y+1, x, y+2, x, y+3])
        elif (rot == 2):
            #  ##X#
            return np.array([x-2,y, x-1, y, x,y, x+1,y])
        elif (rot == 3):
            #  X
            #  #
            #  #
            #  #
            return np.array([x,y, x, y+1, x, y+2, x, y+3])
    elif (piece == 1):
        #Same for all rotations
        #  X#
        #  ##
        return np.array([x,y, x+1, y, x, y+1, x+1, y+1])
    elif (piece == 2):
        
        if (rot == 0):
            # #X#
            #   #
            return np.array([x-1,y,x,y,x+1,y,x+1,y+1])
        elif (rot == 1):
            #  #
            #  X
            # ##
            return np.array([x,y-1,x,y,x,y+1,x-1,y+1])
        elif (rot == 2):
            #  #
            #  #X#
            return np.array([x-1,y-1,x-1,y,x,y,x+1,y])
        elif (rot == 3):
            #  ##
            #  X
            #  #
            return np.array([x,y-1,x+1,y-1,x,y,x,y+1])
    elif (piece == 3):
        if (rot == 0):
            # #X#
            # #
            return np.array([x-1,y,x,y,x+1,y,x-1,y+1])
        elif (rot == 1):
            # ##
            #  X
            #  #
            return np.array([x-1,y-1,x,y-1,x,y,x,y+1])
        elif (rot == 2):
            #   #
            # #X#
            return np.array([x+1,y-1, x+1,y,x,y,x-1,y])
        elif (rot == 3):
            #  #
            #  X
            #  ##
            return np.array([x,y-1,x,y,x,y+1,x+1,y+1])
    elif (piece == 4):
        if (rot == 0 or rot == 2):
            # #X
            #  ##
            return np.array([x-1,y,x,y,x,y+1,x+1,y+1])
        elif (rot == 1 or rot == 3):
            #   #
            #  #X
            #  #
            return np.array([x,y-1,x,y,x-1,y,x-1,y+1])
    
    elif (piece == 5):
        if (rot == 0 or rot == 2):
            #  X#
            # ##
            return np.array([x,y,x+1,y,x-1,y+1,x,y+1])
        elif (rot == 1 or rot == 3):
            #  #
            #  #X
            #   #
            return np.array([x-1,y-1,x-1,y,x,y,x,y+1])
    elif (piece ==6):
        if (rot == 0):
            #  #X#
            #   #
            return np.array([x-1,y,x,y,x+1,y,x,y+1])
        elif (rot == 1):
            #   #
            #  #X
            #   #
            return np.array([x,y-1,x,y,x,y+1,x-1,y])
        elif (rot == 2):
            #   #
            #  #X#
            return np.array([x,y-1,x+1,y,x,y,x-1,y])
        elif (rot == 3):
            #   #
            #   X#
            #   #
            return np.array([x,y-1,x+1,y,x,y+1,x,y])
