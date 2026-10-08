import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from numpy import random as rd
from scipy.linalg import eig
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.linalg import expm 
from numpy import linalg as LA


class MarkovNet:
    def __init__(self,mat=()):
        self.size = np.shape(mat)[0]
        self.R = self.diag_mat(mat)
        (self.eigs, self.u, self.v, self.A) = self.get_eig()
        self.pss = self.v[:,0].real
        #self.get_MFPT_mat()
        #self.t = self.MFPTs+np.diag(self.MFRTs)
         
    def get_eig(self):
        mat = self.R
        size  = np.shape(mat)[0]
        eigs, u,v = eig(mat,left=True)
        # sort eigenvalues and eigenvectors
        sort_indices = eigs.argsort()[::-1]  # return sorted eigenvalues
        eigs = eigs[sort_indices]
        v = v[:,sort_indices] # right eigenvector
        u = u[:,sort_indices].conj() # left eigenvector
        u[:,0] = u[:,0]/u[0,0] # normalize lefteigenvector s.t. <u_0|i> = 1
        uv = np.diag(np.dot(u.T,v)) 
        v = np.array([v[:,i]/uv[i] for i in range(size)]).T # normalize right eigenvector s.t. <u_m|v_n> = \delta_{mn}
        A = np.array([np.dot(np.array([v[:,i]]).T,np.array([u[:,i]])) for i in range(size)]) # {|v_m><u_m|} the rotation matrix
        return eigs,u,v,A
    
    def propag(self,t,method=0):
        # P(i,t|j,0) - for all (i,j)
        if method == 0:
            return np.real([[np.sum(np.exp(self.eigs * t) * self.A[:,i,j]) for j in range(self.size)] for i in range(self.size)])#-np.diag(np.ones(self.size)))
        if method == 1:
            return expm(self.R*t)
        
    def phi(self,i,j):
        return self.R[i,j]*self.pss[j]
    def current(self,i,j):
        return self.phi(i,j)-self.phi(j,i)
    def a(self,i,j):
        return self.phi(i,j)+self.phi(j,i)\
    
    def observable(self,O):
        return np.sum(O*self.pss)

    def VFPT(self,i,j,ret=False):
        if i==j:
            if ret == True:
                vtij = self.MFPT(i,i)* (-2/self.R[i,i]-2/self.pss[i] * np.sum(self.A[1:,i,i]/self.eigs[1:] )) - self.MFPT(i,i)**2 
            else:
                vtij=0
        else:
            vtij =  2*np.sum((self.A[1:,i,i]-self.A[1:,i,j])/self.eigs[1:]**2)/self.pss[i] + (np.sum((self.A[1:,i,i]/self.eigs[1:]))**2-np.sum((self.A[1:,i,j]/self.eigs[1:]))**2)/self.pss[i]**2
        return np.real(vtij) 
    
    def MFPT(self,i,j,ret=False):
        if i == j:
            if ret == True:
                tij = -1/(self.R[i,i]*self.pss[i])
            else:
                tij = 0
        else:
            tij = np.sum((self.A[1:,i,j]-self.A[1:,i,i])/self.eigs[1:])/self.pss[i]
        #if np.imag(tij)!=0:
            #print("wrong!!! non-zero imaginary time")
        return np.real(tij)

    def propag_fast(self,t,i,j):
        # P(i,t|j,0) - directly evaluation
        return np.real(np.sum(np.exp(self.eigs * t) * self.A[:,i,j]))
      
    def Jt(self,t):
        Wt = self.propag(t)/t
        return np.array([[Wt[i,j]*self.pss[j] for j in range(self.size)] for i in range(self.size)])

    def stoc_traj(self,ini=0,nsteps=10,ifplot=False):
        traj = [ini] # The initial state
        t_list = []
        for i in range(0,nsteps):
            rd1,rd2 = np.random.rand(2)
            a_list = np.copy(self.R[:,traj[i]])
            a_list[traj[i]]=0
            a0 = np.sum(a_list)
            # the residence time
            t_list.append(1/a0 * np.log(1/rd1))
            a_acc= np.add.accumulate(a_list)/a0
            # the next state
            next_state = np.argmax(a_acc > rd2)
            traj.append(next_state) 
        if ifplot==True:
            plt.step(np.cumsum(np.concatenate(([0],t_list))),traj,where='post')
        return (np.array(t_list),np.array(traj)) #(residence time along the tranjectory, states along the trajectory)
    
    def get_MFPT_mat(self):
        self.MFPTs = np.array([[self.MFPT(i,j) for j in range(self.size)] for i in range(self.size)])
        self.MFRTs = -(1)/(self.pss*np.diag(self.R))
        return self.MFPTs 
    
    def diag_mat(self,R):
        R = R - np.diag(np.diag(R))
        R = R - np.diag(np.sum(R,axis=0))
        return R
        #self.R = self.R-np.diag(np.diag(self.R))
        #self.R = self.R - np.diag(np.sum(self.R,axis=0))
        
    def Ct(self,t):
        # Cross-correlation
        propag = self.propag(t)
        return np.array([[propag[i,j]*self.pss[j] for j in range(self.size)] for i in range(self.size)])
    
    def Ct_Ratio_fast(self,t,i,j):
        return np.real(np.sum(np.exp(self.eigs * t) * self.A[:,i,j])*self.pss[j]/np.sum(np.exp(self.eigs * t) * self.A[:,j,i])/self.pss[i])
    
    def Ct_fast(self,t,i,j):#The cross-correlation Cij(t) <x_i(t)x_j(0)>
        return self.propag_fast(t,i,j)*self.pss[j]
        
    def J(self,i,j):
        return self.R[i,j]*self.pss[j] - self.R[j,i]*self.pss[i]
        
    def chi(self,t,i,j): 
        # Asymmetry of cross-correlation (Ohga version)
        Ctij = self.Ct_fast(t,i,j)
        Ctji = self.Ct_fast(t,j,i)
        DCtii = self.Ct_fast(t,i,i)-self.Ct_fast(0,i,i)
        DCtjj = self.Ct_fast(t,j,j)-self.Ct_fast(0,j,j)
        return (Ctij - Ctji) / np.sqrt(DCtii * DCtjj) / 2
    
    def chi_new(self,t,i,j):
        # Asymmetry of cross-correlation (Liang version)
        Ctij = self.Ct_fast(t,i,j)
        Ctji = self.Ct_fast(t,j,i)
        return (Ctij-Ctji)/(Ctij+Ctji)
    
    def get_EP(self):
        size = np.shape(self.R)[0]
        Jmat = np.zeros((size,size))
        EP = 0
        for i in range(size):
            for j in range(size):
                if j > i:
                    Jmat[i,j] = self.R[i,j]*self.pss[j]
                    Jmat[j,i] = self.R[j,i]*self.pss[i]
                    if Jmat[i,j]!=0 and Jmat[j,i]!=0:
                        EP += (Jmat[i,j]-Jmat[j,i])*np.log(Jmat[i,j]/Jmat[j,i])
        return EP

    
def shift_mat(diag_matrix):
    # shift the matrix up-left by one entry
    shifted_matrix = np.zeros_like(diag_matrix)
    shifted_matrix[1:, :-1] = diag_matrix[:-1, 1:]
    shifted_matrix[0, :-1] = diag_matrix[-1, 1:]
    shifted_matrix[1:, -1] = diag_matrix[:-1, 0]
    shifted_matrix[0, -1] = diag_matrix[-1, 0]
    return shifted_matrix

def shift_matrix(mat,mov,axis=0):
    """Shift the matrix towards the upper right with periodic boundary conditions"""
    shifted_mat = np.roll(mat, shift=mov, axis=axis)
    return shifted_mat


def eqmat(energies, barriers=None,temperature=1):
    size = len(energies)
    R = np.zeros((size,size))
    if barriers==None:
        barriers = np.ones((size,size))
    for m in range(np.shape(barriers)[0]):
        for n in range(np.shape(barriers)[1]):
            R[m,n] = np.exp(-(np.max((energies[m]-energies[n],0)) + barriers[m,n])/temperature)
    R= R- np.diag(np.sum(R,axis=0))
    return R


def get_rates(n,k0,dk): #randomly generated the transition rates of a unicirectional unicycle network
    return (rd.random(n)*2-1)*dk+np.ones(n)*k0
    #return np.abs(rd.normal(k0,stdk,n))

def tran_mat(rates): #convert transition rates to transition matrix
    return -np.diag(rates) + np.roll(np.diag(rates),-1,axis=0)

def eig_ratio(tran_mat):
    eigvalue,eigenvector = eig(tran_mat)
    
def FPT(rates): #get the first return time (the time of completing one cycle
    return np.power(rates,-1).sum(),np.power(rates,-2).sum()

def get_eig(mat): 
    eigv, u,v = eig(mat,left=True)
    # sort eigenvalues and eigenvectors by magnitude
    sort_indices = eigv.argsort()[::-1]  # indices ordering the eigenvalues
    eigv = eigv[sort_indices]
    v = v[:,sort_indices]
    u = u[:,sort_indices].conj()
    v = v/np.diag(np.dot(u.T,v))
    return eigv,u,v
