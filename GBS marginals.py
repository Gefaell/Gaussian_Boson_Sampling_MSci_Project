'''
Get marginal GBS probabilities attempt
'''
import numpy as np
import pandas as pd
import copy 
#%%

r_k = np.array([
1.6518433645720738,
1.687136454610338,
1.62938385974034,
1.706029877650956,
1.8395638626723685,
1.3943570412472834,
1.4819924169286014,
1.6313980669381827,
1.6559961541267325,
1.3389267197532349,
1.568736620327057,
1.6772334549978614,
1.459031307907052,
1.4124223294979523,
1.3440269631323098,
1.4328684458997072,
1.4675334685180914,
1.6270874674912998,
1.6044404863902908,
1.581538415101846,
1.6519035066626184,
1.5456532234514821,
1.5974577318822245,
1.7043797524114164,
1.7294783286655087])

T_re = pd.read_excel('matrix_re.xlsx', header = None).to_numpy()
T_im = pd.read_excel('matrix_im.xlsx', header = None).to_numpy()
'''
T_re = np.loadtxt('T_re.csv')
#%%
T_real = pd.read_csv('T_re.csv')
'''

T = T_re + T_im * 1j
T = T.T
#%%

def Ch(r):
    return np.array([[np.cosh(r), 0], [0,np.cosh(r)]])

def Sh(r):
    return np.array([[np.sinh(r), 0], [0,np.sinh(r)]])

test = Ch(1)
pad = np.pad(test,(0,3))

S = np.zeros((100,100))



def get_S(r_k):
    Ns = len(r_k)*2
    S_ch = np.zeros((Ns,Ns))
    S_sh = np.zeros((Ns,Ns))
    for count,value in enumerate(r_k):
        ch_mat = Ch(value)
        sh_mat = Sh(value)
        i = 2 * count
        S_ch[i:i+2, i:i+2] = ch_mat
        S_sh[i:i+2, i:i+2] = sh_mat
    S_firstcolumn = np.concatenate((S_ch,S_sh), axis = 0)
    S_secondcolumn = np.concatenate((S_sh,S_ch), axis = 0)
    S = np.concatenate((S_firstcolumn,S_secondcolumn),axis = 1)
    return S

S = get_S(r_k)
    

validate = get_S([1,1])   

def get_sigma_in(S):
    sigma_vac = np.identity(len(S))/2
    return S*sigma_vac*S.T

sigma_in = get_sigma_in(S)
    
def get_sigma(T,sigma_in): #need to add checks for shape length, for now only N=100 will work
    #N=100
    T_len, T_height = np.shape(T)
    TT_len, TT_height = np.shape(T.T)
    first = np.zeros((T_len * 2, T_height * 2))
    second = first.T
    
    first[0:T_len, 0:T_height] = T
    first[T_len:, T_height:] = T.conjugate()
    
    second[0:TT_len, 0:TT_height] = T.T.conjugate()
    second[TT_len:, TT_height:] = T.T
    
    #here = np.identity(T_len*2) - 1/2 * first * second
    return np.identity(T_len*2) - 1/2 * np.dot(first, second) + np.dot(first,np.dot(sigma_in,second)) 


sigma = get_sigma(T,sigma_in)
#%%

def get_reduced(sigma, R):
    '''
    Parameters
    ----------
    s : matrix
        full sigma matrix
    R : 1d array
        marginals considered

    sigma_reduced for the marginal
    -------
    '''
    N = len(sigma)/2
    marginals = copy.deepcopy(R)
    
    for i in R:         
        marginals.append(int(i+N))

    print(marginals)
    sigma_red = np.zeros( (len(marginals),len(marginals)) )
    for r_index,row in enumerate(marginals):
        for c_index,column in enumerate(marginals):
            sigma_red[r_index, c_index] = sigma[row, column]
    print(sigma_red)
    return sigma_red

A= np.array([[1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
    ])

B = array = np.random.randint(10, size=(10, 10))
test_red = get_reduced(B, [0,2,3])

#%%
def p_z(sigma):
    sigma_inv = np.linalg.inv(sigma)
    
    
    O = np.identity(sigma.shape()) - np.linalg.inv
    

    
    