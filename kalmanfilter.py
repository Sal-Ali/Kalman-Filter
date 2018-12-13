import inspect
import sys
import numpy as np
import matplotlib.pyplot as plt

'''
Raise a "not defined" exception as a reminder 
'''

A = np.asmatrix([[1,1],
                [0,1]])
A_T = np.asmatrix([[1,0],
                  [1,1]])
I = np.identity(2)

# process noise covariance
Q = np.asmatrix([[0.0001, 0.00002],
                [0.00002, 0.0001]])

# measurement covariance (noise in measurement)
R = np.asmatrix([[0.02, 0.005],
                [0.005, 0.02]])


u1_saved = 0
u2_saved = 0
x1_saved = 0
x2_saved = 0

X_k_saved = 0
P_kp_saved = 0
count = 0


def _reset():
    global u1_saved, u2_saved, x1_saved, x2_saved, X_k_saved, P_kp_saved, count

    u1_saved = 0
    u2_saved = 0
    X_k_saved = 0
    P_kp_saved = 0
    count = 0
    x1_saved = 0
    x2_saved = 0


def _raise_not_defined():
    print "Method not implemented: %s" % inspect.stack()[1][3]
    sys.exit(1)

'''
Kalman 2D
'''
def kalman2d(data):
    estimated = []
    X_init_x = np.asmatrix([[data[0][2]],
                          [data[0][3]]])
    X_init_u = np.asmatrix([[data[0][0]],
                            [data[0][1]]])
    X_init = np.add(X_init_x, X_init_u)
    #P_init = np.asmatrix([[np.power(data[1][2] - data[0][2], 2), (data[1][2]- data[0][2]) * (data[1][3] - data[0][3])],
    #                     [(data[1][2]- data[0][2]) * (data[1][3] - data[0][3]) ,np.power(data[1][3] - data[0][3], 2)]])
    P_init = I
    #I tested many scalars to I, nothing changes significantly to add accuracy
    P_kp = np.add(A.dot(P_init).dot(A_T), Q)
    #kalman gain
    KG_init = np.divide(P_kp , np.add(P_kp, R))
    #new observation
    Y_k_x = np.matrix([[data[1][2]],
                     [data[1][3]]])
    
    #Y_k = np.multiply(I, Y_k_x)
    #calculate current state estimation
    X_k = X_init + (KG_init * np.subtract(Y_k_x, X_init))
    estimated.append([X_k[0,0], X_k[1,0]])
    #update process covariance matrix
    P_kp = np.multiply(np.subtract(I,KG_init), P_kp)
    
    for i in range(1,len(data)):
        X_init_x = np.asmatrix([[data[i-1][2]],
                              [data[i-1][3]]])
        X_init_u = np.asmatrix([[data[i-1][0]],
                                [data[i-1][1]]])
        X_init = np.add(X_init_x, X_init_u)
  
        P_kp = np.add(A.dot(P_kp).dot(A_T), Q)
        #kalman gain
        KG_init = np.divide(P_kp , np.add(P_kp, R))
        #new observation
        Y_k_x = np.matrix([[data[i][2]],
                         [data[i][3]]])
        #Y_k = np.multiply(I, Y_k_x)
        #calculate current state estimation
        X_k = np.add(X_init, np.multiply(KG_init, np.subtract(Y_k_x,X_init)))
        estimated.append([X_k[0,0], X_k[1,0]])
        #update process covariance matrix
        P_kp = np.multiply(np.subtract(I,KG_init), P_kp)     
        
    #_raise_not_defined()
    return estimated

'''
Plotting
'''
def plot(data, output):
    data = [[d[2], d[3]]for d in data]
    plt.scatter([d[0] for d in data], [d[1] for d in data], color='blue')
    plt.plot([d[0] for d in data], [d[1] for d in data], color='blue')
    plt.scatter([d[0] for d in output], [d[1] for d in output], color='r')
    plt.plot([d[0] for d in output], [d[1] for d in output], color='r')
    
    plt.show()
    

'''
Kalman 2D 
'''
def kalman2d_shoot(ux, uy, ox, oy, reset=False):
    global u1_saved, u2_saved, x1_saved, x2_saved, X_k_saved, P_kp_saved, count

    decision = (0, 0, False)
    if reset is True:
        _reset()
        P_kp_saved = I
        
    else:
        x_init_x = np.asmatrix([[x1_saved],[x2_saved]])
        x_init_u = np.asmatrix([[u1_saved],[u2_saved]])
        x_init = np.add(x_init_x, x_init_u)
        # P_kp_saved = np.add(A.dot(P_kp_saved).dot(A_T), Q)
        P_kp_saved = np.add(P_kp_saved, Q)
        KG_init = np.divide(P_kp_saved , np.add(P_kp_saved, R))
        y_k = np.asmatrix([[ox],[oy]])
        X_k_saved = np.add(x_init, np.multiply(KG_init, np.subtract(y_k,x_init)))
        P_kp_saved = np.multiply(np.subtract(I,KG_init), P_kp_saved)  
    count += 1   
    u1_saved = ux
    u2_saved = uy
    x1_saved = ox
    x2_saved = oy 
 
    if count == 50:
        count = 0 
        decision = (X_k_saved[0,0], X_k_saved[1,0], True)
        
    #_raise_not_defined()
    return decision

'''
Kalman 2D  final
'''
def kalman2d_adv_shoot(ux, uy, ox, oy, reset=False):
    global u1_saved, u2_saved, x1_saved, x2_saved, X_k_saved, P_kp_saved, count

    decision = (0, 0, False)
    if reset is True:
        _reset()
        P_kp_saved = I
        
    else:
        x_init_x = np.asmatrix([[x1_saved],[x2_saved]])
        x_init_u = np.asmatrix([[u1_saved],[u2_saved]])
        x_init = np.add(x_init_x, x_init_u)
        # P_kp_saved = np.add(A.dot(P_kp_saved).dot(A_T), Q)
        P_kp_saved = np.add(P_kp_saved, Q)
        KG_init = np.divide(P_kp_saved , np.add(P_kp_saved, R))
        y_k = np.asmatrix([[ox],[oy]])
        X_k_saved = np.add(x_init, np.multiply(KG_init, np.subtract(y_k,x_init)))
        P_kp_saved = np.multiply(np.subtract(I,KG_init), P_kp_saved)  
    count += 1   
    u1_saved = ux
    u2_saved = uy
    x1_saved = ox
    x2_saved = oy 
 
    if count == 50:
        count = 0 
        decision = (X_k_saved[0,0], X_k_saved[1,0], True)
    return decision


