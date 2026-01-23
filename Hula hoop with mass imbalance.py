import numpy as np
import os
try:
    import scipy.io as sio  # type: ignore
except Exception:  # pragma: no cover
    sio = None
from datetime import datetime
import shutil
import json


# creating custom exceptions
class MaxNewtonIterAttainedError(Exception):
    """This exception is raised when the maximum number of Newton iterations is attained
      whilst the iterations have not yet converged and the solution was not yet obtained."""
    def __init__(self, message="This exception is raised when the maximum number of Newton iterations is attained."):
        self.message = message
        super().__init__(self.message)

class NoOpenContactError(Exception):
    """Contact is not open."""
    def __init__(self, message="This exception is raised when the contact is not open."):
        self.message = message
        super().__init__(self.message)

class RhoInfInfiniteLoop(Exception):
    """This exception is raised when we have possibly entered in an infinite loop through updating rho_inf."""
    def __init__(self, message="This exception is raised when we have possibly entered in an infinite loop through updating rho_inf."):
        self.message = message
        super().__init__(self.message)

class MaxHoursAttained(Exception):
    """This exception is raised when the maximum number of run hours specified by the user is exceeded."""
    def __init__(self, message="This exception is raised when the maximum run time is exceeded."):
        self.message = message
        super().__init__(self.message)

class JacobianBlowingUpError(Exception):
    """This exception is raised when the Jacobian is blowing up."""
    def __init__(self, message="This exception is raised when the Jacobian is blowing up."):
        self.message = message
        super().__init__(self.message)

class Simulation:
    def __init__(
        self,
        ntime=5,
        mu_s=10**9,
        mu_k=0.3,
        eN=0,
        eF=0,
        R=0.1,
        m_hoop=1.0,
        m_imbalance=0.1,
        theta0=0.0,
        omega0=10.0,
        start_in_contact=True,
        enforce_gap_distance=True,
        y0=0.0,
        x0=0.0,
    ):
        # path for outputs
        # Generate timestamp
        timestamp = datetime.now().strftime("hula_hoop_%Y-%m-%d_%H-%M-%S")
        outputs_dir = f"outputs/{timestamp}"
        self.output_path = os.path.join(os.getcwd(), outputs_dir)  # Output path
        os.makedirs(self.output_path, exist_ok=True)

        # Path to the current file
        current_file = os.path.realpath(__file__)
        # Copy the file
        shutil.copy2(current_file, self.output_path)

        # friction coefficients
        self.mu_s = mu_s    # Static friction coefficient
        self.mu_k = mu_k    # Kinetic friction coefficient
        # restitution coefficients
        self.eN = eN        # normal coefficient of restitution
        self.eF = eF        # friction coefficient of restitution
        # nondimensionalization parameters
        l_nd = 1       # m, length nondimensionalization paramter
        m_nd = 1       # kg, mass nondimensionalization parameter
        a_nd = 9.81    # m/(s**2), acceleration nondimensionalization parameter
        t_nd = np.sqrt(l_nd/a_nd)   # s, time nondimensionalization parameter
        # simulation (time) parameters
        self.dtime = 2e-3/t_nd # time step duration
        self.ntime = ntime           # number of iterations
        self.tf = self.ntime*self.dtime            # final time
        self.t = np.linspace(0,self.tf,self.ntime) # time array
        # hoop properties (planar rigid body with COM offset from geometric center)
        # - geometric radius: R
        # - hoop mass: m_hoop (thin ring)
        # - imbalance: m_imbalance placed on the rim at +x_body
        self.R = float(R) / l_nd
        self.m_hoop = float(m_hoop) / m_nd
        self.m_imbalance = float(m_imbalance) / m_nd
        self.m = self.m_hoop + self.m_imbalance

        # COM offset from geometric center G along +x_body
        self.d = (self.m_imbalance * self.R) / self.m

        # Inertia about COM (z-axis)
        # thin ring about G: I_h_G = m_h R^2
        # shift ring to COM: I_h_C = I_h_G + m_h d^2
        # point mass at rim relative to COM: (R - d)
        I_h_C = self.m_hoop * self.R**2 + self.m_hoop * self.d**2
        I_p_C = self.m_imbalance * (self.R - self.d) ** 2
        self.ecc = self.m_imbalance /self.m
        self.I = self.m*self.R**2*(1-self.ecc**2)
        # nondimensional constants
        self.gr = 9.81/a_nd    # gravitational acceleration
        # total degrees of freedom
        self.ndof = 3       # x,y, theta of hoop (at COM)
        # constraint count
        self.ng = 0          # number of constraints at position level
        self.ngamma = 0      # number of constraints at velocity level
        # Ground contact model is always present; this flag only controls whether
        # we auto-project the *initial* configuration to satisfy gN = 0.
        self.enforce_gap_distance = bool(enforce_gap_distance)
        self.nN = 1          # number of gap distance constraints
        self.nF = 1          # number of friction constraints
        self.gammaF_lim = np.array([[0]])    # connectivities of friction and normal forces
        self.nX = 3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+2*self.nF     # total number of constraints with their derivative
        # generalized alpha parameters
        self.MAXITERn = 20
        self.MAXITERn_initial = self.MAXITERn   # saving initial value of MAXITERn
        self.r = 0.3
        self.rho_inf = 0.5
        self.rho_infinity_initial = self.rho_inf
        # eq. 72
        self.alpha_m = (2*self.rho_inf-1)/(self.rho_inf+1)
        self.alpha_f = self.rho_inf/(self.rho_inf+1)
        self.gama = 0.5+self.alpha_f-self.alpha_m
        self.beta = 0.25*(0.5+self.gama)**2
        self.tol_n = 1.0e-6     # error tolerance
        # Note: Mass matrix is time-varying and computed in get_R() method
        # applied forces (weight)
        self.force = np.array([[0,-self.m*self.gr,0]])
        self.q_save = np.zeros((self.ndof,self.ntime))
        self.u_save = np.zeros((self.ndof,self.ntime))
        self.X_save = np.zeros((self.nX,self.ntime))
        self.gNdot_save = np.zeros((self.nN,self.ntime))
        self.gammaF_save = np.zeros((self.nF,self.ntime))
        self.AV_save = np.zeros((self.ndof+self.nN+self.nF,self.ntime))
        self.contacts_save = np.zeros((5*self.nN,self.ntime))

        # initial position (COM coordinates)
        # If starting in contact, enforce gN = 0:
        # gN = y_G - R = (y - d*sin(theta)) - R = 0 => y = R + d*sin(theta)
        theta0 = float(theta0)
        self.start_in_contact = bool(start_in_contact)
        self.x0_input = float(x0)
        self.y0_input = float(y0)
        self.theta0_input = float(theta0)
        self.omega0_input = float(omega0)
        y0_nd = y0 + self.R + self.d * np.sin(theta0)
        x0_nd = x0 + self.d * np.cos(theta0)
        q0 = np.array([float(x0_nd), float(y0_nd), theta0])
        self.q_save[:,0] = q0
        # initial velocity
        # User-specified initial angular velocity (about +z)
        u0 = np.array([0.0, 0.0, float(omega0)])
        self.u_save[:,0] = u0
    
        # creating an output file f to log major happenings
        self.f = open(f"{self.output_path}/log_file.txt",'a')

        # Save metadata once for visualization/reproducibility
        self._save_metadata()

    def _save_metadata(self):
        """Save run parameters and derived quantities for post-processing."""
        params = {
            "dtime": float(self.dtime),
            "ntime": int(self.ntime),
            "R": float(self.R),
            "m_hoop": float(self.m_hoop),
            "m_imbalance": float(self.m_imbalance),
            "m_total": float(self.m),
            "d": float(self.d),
            "I": float(self.I),
            "mu_s": float(self.mu_s),
            "mu_k": float(self.mu_k),
            "eN": float(self.eN),
            "eF": float(self.eF),
            "enforce_gap_distance": bool(self.enforce_gap_distance),
            "start_in_contact": bool(self.start_in_contact),
            "x0_input": float(self.x0_input),
            "y0_input": None if self.y0_input is None else float(self.y0_input),
            "theta0_input": float(self.theta0_input),
            "omega0_input": float(self.omega0_input),
            "q0": [float(self.q_save[0,0]), float(self.q_save[1,0]), float(self.q_save[2,0])],
            "u0": [float(self.u_save[0,0]), float(self.u_save[1,0]), float(self.u_save[2,0])],
            "rho_inf": float(self.rho_inf),
            "alpha_m": float(self.alpha_m),
            "alpha_f": float(self.alpha_f),
            "gamma": float(self.gama),
            "beta": float(self.beta),
        }
        with open(os.path.join(self.output_path, "params.json"), "w", encoding="utf-8") as fp:
            json.dump(params, fp, indent=2)

    def save_arrays(self):
        """Saving arrays."""
        if sio is not None:
            file_name_q = str(f'{self.output_path}/q.mat')
            sio.savemat(file_name_q,dict(q=self.q_save))

            file_name_u = str(f'{self.output_path}/u.mat')
            sio.savemat(file_name_u,dict(u=self.u_save))

            file_name_x_save = str(f'{self.output_path}/x_save.mat')
            sio.savemat(file_name_x_save,dict(X=self.X_save))

            file_name_contacts = str(f'{self.output_path}/contacts.mat')
            sio.savemat(file_name_contacts,dict(contacts=self.contacts_save))
        else:
            try:
                self.f.write("\n  SciPy unavailable; skipping .mat outputs")
            except Exception:
                pass

        np.save(f'{self.output_path}/q_save.npy', self.q_save)
        np.save(f'{self.output_path}/u_save.npy', self.u_save)
        np.save(f'{self.output_path}/X_save.npy', self.X_save)
        np.save(f'{self.output_path}/gNdot_save.npy', self.gNdot_save)
        np.save(f'{self.output_path}/gammaF_save.npy', self.gammaF_save)
        np.save(f'{self.output_path}/AV_save.npy', self.AV_save)
        return
 
    def get_R(self,iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,*index_sets):
        """Calculates the residual."""

        [prev_a,_,_,_,_,_,_,_,_,_,prev_lambdaN,_,prev_lambdaF] = self.get_X_components(prev_X)
        [a,U,Q,Kappa_g,Lambda_g,lambda_g,Lambda_gamma,lambda_gamma,
            KappaN,LambdaN,lambdaN,LambdaF,lambdaF] = self.get_X_components(X)
        
        # AV - Auxiliary Variables [abar, lambdaNbar, lambdaFbar]
        prev_abar = prev_AV[0:self.ndof]
        prev_lambdaNbar = prev_AV[self.ndof:self.ndof+self.nN]
        prev_lambdaFbar = prev_AV[self.ndof+self.nN:self.ndof+self.nN+self.nF]

        # auxiliary variables update
        # eq. 49
        abar = (self.alpha_f*prev_a+(1-self.alpha_f)*a-self.alpha_m*prev_abar)/(1-self.alpha_m)
        # eq. 96
        lambdaNbar = (self.alpha_f*prev_lambdaN+(1-self.alpha_f)*lambdaN-self.alpha_m*prev_lambdaNbar)/(1-self.alpha_m)
        # eq. 114
        lambdaFbar = (self.alpha_f*prev_lambdaF+(1-self.alpha_f)*lambdaF-self.alpha_m*prev_lambdaFbar)/(1-self.alpha_m)

        AV = np.concatenate((abar,lambdaNbar,lambdaFbar),axis=None)

        # velocity update (73)
        u = prev_u+self.dtime*((1-self.gama)*prev_abar+self.gama*abar)+U
        # position update (73)
        q = prev_q+self.dtime*prev_u+self.dtime**2/2*((1-2*self.beta)*prev_abar+2*self.beta*abar)+Q

        # Mass matrix (time-varying, depends on q[2]=theta)
        theta = q[2]
        M = np.array([[self.m, 0, -self.m*self.ecc*self.R*np.sin(theta)],
                      [0, self.m, self.m*self.ecc*self.R*np.cos(theta)],
                      [-self.m*self.ecc*self.R*np.cos(theta), self.m*self.ecc*self.R*np.sin(theta), self.m*self.R**2]])

        # bilateral constraints at position level
        g = np.zeros((self.ng))
        gdot = np.zeros((self.ng))
        gddot = np.zeros((self.ng))
        Wg = np.zeros((self.ndof,self.ng))

        # bilateral constraints at velocity level
        gamma = np.zeros((self.ngamma))
        gammadot = np.zeros((self.ngamma))
        Wgamma = np.zeros((self.ndof,self.ngamma))

        # normal gap distance and slip speed constraints
        gN = np.array([q[1]-self.R - self.d * np.sin(q[2])])
        gNdot = np.array([u[1] - self.d * np.cos(q[2]) * u[2]])
        gNddot = np.array([a[1] - self.d * np.cos(q[2]) * a[2] + self.d * np.sin(q[2]) * u[2]**2])
        WN = np.array([[0,1,-self.d * np.cos(q[2])]]).T
        gammaF = np.array([u[0]+(self.R + self.d * np.sin(q[2]))*u[2]])
        gammaFdot = np.array([a[0]+(self.R + self.d * np.sin(q[2]))*a[2] + self.d * np.cos(q[2]) * u[2]**2])
        WF = np.array([[1,0,self.R + self.d * np.sin(q[2])]]).T

        theta = q[2]
        omega = u[2]

        # eq. 44
        ksiN = gNdot+self.eN*prev_gNdot
        # discrete normal percussion eq. 95
        PN = LambdaN+self.dtime*((1-self.gama)*prev_lambdaNbar+self.gama*lambdaNbar)
        # eq. 102
        Kappa_hatN = KappaN+self.dtime**2/2*((1-2*self.beta)*prev_lambdaNbar+2*self.beta*lambdaNbar)

        # eq. 48 (use friction restitution coefficient)
        ksiF = gammaF+self.eF*prev_gammaF
        # eq. 113
        PF = LambdaF+self.dtime*((1-self.gama)*prev_lambdaFbar+self.gama*lambdaFbar)
            
        Rs = np.concatenate(([M@a-self.force-Wg@lambda_g-Wgamma@lambda_gamma-WN@lambdaN-WF@lambdaF],
                [M@U-Wg@Lambda_g-Wgamma@Lambda_gamma-WN@LambdaN-WF@LambdaF],
                [M@Q-Wg@Kappa_g-WN@KappaN-self.dtime/2*(Wgamma@Lambda_gamma+WF@LambdaF)],
                g,
                gdot,
                gddot,
                gamma,
                gammadot),axis=None)
        
        # Contact residual Rc
        R_KappaN = np.zeros(self.nN)   # (129)
        R_LambdaN = np.zeros(self.nN)
        R_lambdaN = np.zeros(self.nN)
        R_LambdaF = np.zeros(self.nF)  # (138)
        R_lambdaF = np.zeros(self.nF)  # (142)

        if index_sets == ():
            A = np.zeros(self.nN, dtype=int)
            B = np.zeros(self.nN, dtype=int)
            C = np.zeros(self.nN, dtype=int)
            D = np.zeros(self.nN, dtype=int)
            E = np.zeros(self.nN, dtype=int)

            for i in range(self.nN):
                # check for contact if blocks are not horizontally detached
                if self.r*gN[i] - Kappa_hatN[i] <=0:
                    A[i] = 1
                    if np.linalg.norm(self.r*ksiF[self.gammaF_lim[i,:]]-PF[self.gammaF_lim[i,:]])<=self.mu_s*(PN[i]):
                        # D-stick
                        D[i] = 1
                        if np.linalg.norm(self.r*gammaFdot[self.gammaF_lim[i,:]]-lambdaF[self.gammaF_lim[i,:]])<=self.mu_s*(lambdaN[i]):
                            # E-stick
                            E[i] = 1
                    if self.r*ksiN[i]-PN[i] <= 0:
                        B[i] = 1
                        if self.r*gNddot[i]-lambdaN[i] <= 0:
                            C[i] = 1
        else:
            A = index_sets[0]
            B = index_sets[1]
            C = index_sets[2]
            D = index_sets[3]
            E = index_sets[4]

        # calculating contact residual
        for k in range(self.nN):
            if A[k]:
                R_KappaN[k] = gN[k]
                if D[k]:
                    R_LambdaF[self.gammaF_lim[k,:]] = ksiF[self.gammaF_lim[k,:]]
                    if E[k]:
                        R_lambdaF[self.gammaF_lim[k,:]] = gammaFdot[self.gammaF_lim[k,:]]
                    else:
                        R_lambdaF[self.gammaF_lim[k,:]] = lambdaF[self.gammaF_lim[k,:]]+self.mu_k*lambdaN[k]*np.sign(gammaFdot[self.gammaF_lim[k,:]])                    
                else:
                    R_LambdaF[self.gammaF_lim[k,:]] = PF[self.gammaF_lim[k,:]]+self.mu_k*PN[k]*np.sign(ksiF[self.gammaF_lim[k,:]])
                    R_lambdaF[self.gammaF_lim[k,:]] = lambdaF[self.gammaF_lim[k,:]]+self.mu_k*lambdaN[k]*np.sign(gammaF[self.gammaF_lim[k,:]])
            else:
                R_KappaN[k] = Kappa_hatN[k]
                R_LambdaF[self.gammaF_lim[k,:]] = PF[self.gammaF_lim[k,:]]
                R_lambdaF[self.gammaF_lim[k,:]] = lambdaF[self.gammaF_lim[k,:]]
            # (132)
            if B[k]:
                R_LambdaN[k] = ksiN[k]
            else:
                R_LambdaN[k] = PN[k]
            # (135)
            if C[k]:
                R_lambdaN[k] = gNddot[k]
            else:
                R_lambdaN[k] = lambdaN[k]


        Rc = np.concatenate((R_KappaN, R_LambdaN, R_lambdaN, R_LambdaF, R_lambdaF),axis=None)
        
        R = np.concatenate([Rs, Rc],axis=None)


        if index_sets == ():
            # in this case, get_R is called to calculate the actual residual, not as part of calculating the Jacobian
            # print contact region indicators once per residual evaluation
            print(f"A={A}")
            print(f"B={B}")
            print(f"C={C}")
            print(f"D={D}")
            print(f"E={E}")
            return R, AV, q, u, gNdot, gammaF, A, B, C, D, E
        else:
            # in this case, get_R is called as part of calculating the Jacobian for fixed contact regions
            return R, AV, q, u, gNdot, gammaF

    def get_R_J(self,iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF):
        '''Calculate the Jacobian manually.'''

        epsilon = 1e-6
        R, AV, q, u, gNdot, gammaF, A, B, C, D, E = self.get_R(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)
        contacts_nu = np.concatenate((A,B,C,D,E),axis=None)

        # Initializing the Jacobian
        J = np.zeros((self.nX,self.nX))
        I = np.identity(self.nX)

        # Constructing the Jacobian column by column
        for i in range(self.nX):
            # print(i)
            R_plus_epsilon,_,_,_,_,_ = self.get_R(iter,X+epsilon*I[:,i],prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF, A, B, C, D, E)
            J[:,i] = (R_plus_epsilon-R)/epsilon

        return R, AV, q, u, gNdot, gammaF, J, contacts_nu

    def update(self,iter,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF):
        """Takes components at time t and return values at time t+dt"""

        nu = 0
        
        X = prev_X
        R, AV, q, u, gNdot, gammaF, J, contacts_nu = self.get_R_J(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)

        contacts = np.zeros((self.MAXITERn+1,3*self.nN+2*self.nN),dtype=int)
        contacts[nu,:] = contacts_nu
        self.contacts_save[:,iter] = contacts_nu

        norm_R = np.linalg.norm(R,np.inf)
        print(f"norm(R) = {norm_R}")

        # try:

        while np.abs(np.linalg.norm(R,np.inf))>self.tol_n and nu<self.MAXITERn:
            # Newton Update
            X = X-np.linalg.solve(J,R)
            # Calculate new EOM and residual
            nu = nu+1

            R, AV, q, u, gNdot, gammaF, J, contacts_nu = self.get_R_J(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)

            contacts[nu,:] = contacts_nu
            self.contacts_save[:,iter] = contacts_nu
                
            norm_R = np.linalg.norm(R,np.inf)
            print(f"nu = {nu}")
            print(f"norm(R) = {norm_R}")

        return X,AV,q,u,gNdot,gammaF
                
    def update_rho_inf(self):
        '''Update the numerical parameter rho_inf.'''
        self.rho_inf = self.rho_inf+0.05  #0.01
        print(self.rho_inf)
        self.f.write(f"  Updating rho_inf to {self.rho_inf}")
        if np.abs(self.rho_inf - self.rho_infinity_initial) < 0.001:
            print("possibility of infinite loop")
            self.f.write(f"  Raising RhoInfInfiniteLoop error")
            raise RhoInfInfiniteLoop
        if self.rho_inf > 1.001:
            self.rho_inf = 0
        # eq. 72
        self.alpha_m = (2*self.rho_inf-1)/(self.rho_inf+1)
        self.alpha_f = self.rho_inf/(self.rho_inf+1)
        self.gama = 0.5+self.alpha_f-self.alpha_m
        self.beta = 0.25*(0.5+self.gama)**2

    def get_X_components(self,X):
        '''Getting the components of the array X.'''
        a = X[0:self.ndof]
        U = X[self.ndof:2*self.ndof]
        Q = X[2*self.ndof:3*self.ndof]
        Kappa_g = X[3*self.ndof:3*self.ndof+self.ng]
        Lambda_g = X[3*self.ndof+self.ng:3*self.ndof+2*self.ng]
        lambda_g = X[3*self.ndof+2*self.ng:3*self.ndof+3*self.ng]
        Lambda_gamma = X[3*self.ndof+3*self.ng:3*self.ndof+3*self.ng+self.ngamma]
        lambda_gamma = X[3*self.ndof+3*self.ng+self.ngamma:3*self.ndof+3*self.ng+2*self.ngamma]
        Kappa_N = X[3*self.ndof+3*self.ng+2*self.ngamma:3*self.ndof+3*self.ng+2*self.ngamma+self.nN]
        Lambda_N = X[3*self.ndof+3*self.ng+2*self.ngamma+self.nN:3*self.ndof+3*self.ng+2*self.ngamma+2*self.nN]
        lambda_N = X[3*self.ndof+3*self.ng+2*self.ngamma+2*self.nN:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN]
        Lambda_F = X[3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+self.nF]
        lambda_F = X[3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+self.nF:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+2*self.nF]
        return a,U,Q,Kappa_g,Lambda_g,lambda_g,Lambda_gamma,lambda_gamma,\
            Kappa_N,Lambda_N,lambda_N,Lambda_F,lambda_F

    def time_update(self, iter):

        prev_X = self.X_save[:,iter-1]
        prev_AV = self.AV_save[:,iter-1]
        prev_q = self.q_save[:,iter-1]
        prev_u = self.u_save[:,iter-1]
        prev_gNdot = self.gNdot_save[:,iter-1]
        prev_gammaF = self.gammaF_save[:,iter-1]

        # try:
        X,AV,q,u,gNdot,gammaF = self.update(iter,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)

        self.q_save[:,iter] = q
        self.u_save[:,iter] = u
        self.X_save[:,iter] = X
        self.gNdot_save[:,iter] = gNdot
        self.gammaF_save[:,iter] = gammaF
        self.AV_save[:,iter] = AV
        self.save_arrays()

        return
        
    def solve(self):
        iter = 1
        while iter < self.ntime:
            self.time_update(iter)
            iter += 1

        try:
            self.f.close()
        except Exception:
            pass

if __name__ == "__main__":
    # Example: hoop with rim mass imbalance, started in contact with a prescribed spin.
    test = Simulation(
        ntime=1000,
        mu_s=10**9,
        mu_k=0.27,
        eN=0.5,
        eF=0.0,
        R=0.1,
        m_hoop=1,
        m_imbalance=0.67,
        theta0=np.pi,
        omega0=0.5, # CCW
        start_in_contact=True,
        # If you don't want initial projection to gN=0, pass an explicit y0.
        enforce_gap_distance=True,
        y0=0, # gap at t=0
        x0=0.0,
    )
    test.solve()