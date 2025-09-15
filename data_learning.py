'''
Gaussian process learning
Please find the corresponding article for more details about the algorithm
'''
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.numpy.linalg import solve
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
from sklearn.metrics import mean_squared_error

import os
import sys
sys.path.append(os.getcwd())
np.random.seed(3)


class GaussianMixtureModel():
    """Representation of a Gaussian Mixture Model probability distribution. The class allows for the estimation of the 
    parameters of a GMM, specifically in the case of learning from demonstration. The class is based on sklearn's GMM 
    implementation, expanding it to implement Gaussian Mixture Regression.

    Parameters
    ----------
    n_components : int, default = 10
        Number of Gaussian components of the model.
    n_demos : int, default = 5
        Number of demonstrations in the training dataset.
    diag_reg_factor : float, default = 1e-6
        Non negative regularization factor added to the diagonal of the covariances to ensure they are positive.
    """

    def __init__(self,
                 n_components: int = 10,
                 n_demos: int = 5,
                 diag_reg_factor: float = 1e-4) -> None:
        self.n_components = n_components
        self.n_demos = n_demos
        self.diag_reg_factor = diag_reg_factor
        self.model = GaussianMixture(
            n_components=n_components, reg_covar=diag_reg_factor, random_state=420)
        self.logger = logging.getLogger(__name__)

    def fit(self, data: np.ndarray) -> None:
        """Wrapper around sklearn's GMM fit implementation.

        Parameters
        ----------
        data : np.ndarray
            The dataset to fit the model on.
        """
        self.n_features = data.shape[1]
        self.model.fit(data)
        self.logger.info("GMM fit done.")
        self.priors = self.model.weights_
        self.means = self.model.means_.T # transpose to have shape (n_features, n_samples)
        self.covariances = np.transpose(self.model.covariances_, (1, 2, 0)) # transpose to have shape (n_features, n_features, n_samples)

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use Gaussian Mixture Regression to predict mean and covariance of the given inputs

        Parameters
        ----------
        data : ArrayLike of shape (n_input_features, n_samples)

        Returns
        -------
        means : ArrayLike of shape (n_output_features, n_samples)
            The mean vectors associated to each input point.
        covariances : ArrayLike of shape (n_output_features, n_output_features, n_samples)
            The covariance matrices associated to each input point.
        """
        # Dimensionality of the inputs, number of points
        I, N = data.shape
        # Dimensionality of the outputs
        O = self.n_features - I
        diag_reg_factor = np.eye(O)*self.diag_reg_factor
        # Initialize needed arrays
        mu_tmp = np.zeros((O, self.n_components))
        means = np.zeros((O, N))
        covariances = np.zeros((O, O, N))
        H = np.zeros((self.n_components, N))
        for t in range(N):
            # Activation weight
            for i in range(self.n_components):
                mu = self.means[:I, i]
                sigma = self.covariances[:I, :I, i]
                dist = multivariate_normal(mu, sigma)
                H[i, t] = self.priors[i] * dist.pdf(data[:,t])
            H[:, t] /= np.sum(H[:, t] + REALMIN)
            # Conditional means
            for i in range(self.n_components):
                sigma_tmp = self.covariances[I:, :I, i]@inv(self.covariances[:I, :I, i])
                mu_tmp[:, i] = self.means[I:, i] + \
                    sigma_tmp@(data[:, t]-self.means[:I, i])
                means[:, t] += H[i, t]*mu_tmp[:, i]
            # Conditional covariances
            for i in range(self.n_components):
                sigma_tmp = self.covariances[I:, I:, i] - \
                    self.covariances[I:, :I, i]@inv(
                        self.covariances[:I, :I, i])@self.covariances[:I, I:, i]
                covariances[:, :, t] += H[i, t] * \
                    (sigma_tmp + np.outer(mu_tmp[:, i], mu_tmp[:, i]))
            covariances[:, :, t] += diag_reg_factor - \
                np.outer(means[:, t], means[:, t])
        self.logger.info("GMR done.")
        return means, covariances

    def bic(self, data: np.ndarray) -> float:
        """Wrapper around sklearn's GMM BIC function.

        Parameters
        ----------
        data : np.ndarray
            The data to evaluate the BIC in.

        Returns
        -------
        float
            The computed BIC. The lower the better.
        """
        return self.model.bic(data)


class Ours:
    def __init__(self, X, y, X_, y_, observation_noise=0.1, dt = 0.005):
        '''
        :param X: Original input set
        :param y: Original output set
        :param X_: Via-points input set
        :param y_: Via-points output set
        :param observation_noise: Observation noise for y
        '''
        self.X_total = np.vstack((X, X_))#(1001,1)
        # self.y_total = np.vstack((y.reshape(-1, 1), y_.reshape(-1, 1))).reshape(-1)#(1001,)
        self.y_total = np.vstack((y, y_))#.reshape(-1)#(1001,)
        self.X = X#(1000,1)
        self.X_ = X_#(1,1)
        self.y = y#(1000,)
        self.dim = self.y.shape[1]
        self.y_ = y_#(1,)
        self.input_dim = np.shape(self.X_total)[1]# 1
        self.input_num = np.shape(self.X_total)[0]#1001
        self.via_points_num = np.shape(X_)[0]# 1
        self.observation_noise = observation_noise# 1
        self.dt = dt

        # Initialize the parameters
        self.param = self.init_random_param()
        
        # p = [self.y_]
        # var = np.eye(2)*1e-6
        # self.set_waypoint( self.X_[0], p, [var])
    
    def GMM(self, time, pos, t):
         X = np.vstack((time.T, pos.T)).T
         gmm = GaussianMixtureModel(n_components=8, n_demos=5,diag_reg_factor=1e-6)
         gmm.fit(X)
         mu, sigma = gmm.predict(t)
         return mu.T
     
    def init_random_param(self):
        '''
        Initialize the hyper-parameters of GP-MP
        :return: Initial hyper-parameters
        '''
        kern_length_scale = 0.1 * np.random.normal(size=self.input_dim) + 1#(1,)
        kern_noise = 1 * np.random.normal(size=1)#(1,) 随机噪声
        print("初始化参数:", kern_noise, kern_length_scale)
        return np.hstack((kern_noise, kern_length_scale))

    def mse(self, x, traj_set):
        # mse_ = mean_squared_error(x, traj_set)
        mse_ = (x - traj_set)**2 
        mse_ = np.sum(mse_)/traj_set.shape[0]
        # mse_ = np.mean(mse_)
        return mse_
    
    def build_objective(self, param):
        '''
        Compute the objective function (log pdf)
        :param param: Hyper-parameters of GP-MP
        :return: Value of the obj function
        '''
        cov_y_y_total = self.rbf(self.X_total, self.X_total, param)#K(t_hat, t_hat, sita)
        variance_matrix = np.zeros((self.input_num, self.input_num)) * 1.0
        variance_matrix[0:(self.input_num - self.via_points_num), 0:(self.input_num - self.via_points_num)] = \
            self.observation_noise**2 * np.eye(self.input_num - self.via_points_num)#主要生成噪声
        cov_y_y_total = cov_y_y_total + variance_matrix
        beta = solve(cov_y_y_total, self.mu)#求解线性方程的解
        mean_outputs = np.dot(cov_y_y_total.T, beta)
        out = 0
        for i in range(self.dim):
                out += - mvn.logpdf(self.mu[:,i], np.zeros(self.input_num), cov_y_y_total)
        out = self.mse(mean_outputs, self.mu) + out
        return out

    def train(self, traject):
        def KL_mse(xi, traj_set) -> float:
            mse = []
            for i in range(len(traj_set)):
                mm = traj_set[i]
                mse_ = mean_squared_error(mm, xi.T[:2,:])
                mse.append(mse_)
            mse = np.array(mse)
            # mse = np.sum(mse) / len(traj_set)
            mse = np.mean(mse)
            # print("KMP_MSE:", mse) 
            return mse
        def cons_f(param):
            '''
            Constrained function, see Eq.(20) of the article
            :param param: Hyper-parameters of GP-MP
            :return: Value of the constrained function
            '''
            delta = 1e-10
            cov_y_y_ = self.rbf(self.X_, self.X_, param)
            min_eigen = np.min(np.linalg.eigvals(cov_y_y_))
            return min_eigen - delta

        # Using "trust-constr" approach to minimize the obj
        nonlinear_constraint = NonlinearConstraint(cons_f, 0.0, np.inf, jac='2-point', hess=BFGS())#
        x_gmr = self.dt*np.arange(int(self.X_total.shape[0])).reshape(1,-1)
        self.mu = self.GMM(self.X,self.y,x_gmr)
        result = minimize(value_and_grad(self.build_objective), self.param, method='trust-constr', jac=True,
                          options={'disp': True, 'maxiter': 500, 'xtol': 1e-50, 'gtol': 1e-20},
                          constraints=[nonlinear_constraint], callback=self.callback)
        # Pre-computation for prediction
        self.param = result.x
        variance_matrix = np.zeros((self.input_num, self.input_num)) * 1.0
        variance_matrix[0:(self.input_num - self.via_points_num), 0:(self.input_num - self.via_points_num)] = \
            self.observation_noise ** 2 * np.eye(self.input_num - self.via_points_num)
        self.cov_y_y_total = self.rbf(self.X_total, self.X_total, self.param) + variance_matrix
        self.beta = solve(self.cov_y_y_total, self.y_total)
        self.inv_cov_y_y_total = solve(self.cov_y_y_total, np.eye(self.input_num))

    def rbf(self, x, x_, param):
        '''
        Interface to compute the Variance matrix (vector) of GP,
        :param x: Input 1
        :param x_: Input 2
        :param param: Hyper-parameters of GP-MP
        :return: Variance matrix (vector)
        '''
        kern_noise = param[0]
        sqrt_kern_length_scale = param[1:]
        diffs = np.expand_dims(x / sqrt_kern_length_scale, 1) - np.expand_dims(x_ / sqrt_kern_length_scale, 0)
        return kern_noise**2 * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))
    

    def predict_determined_input(self, x):
        '''
        Compute the mean and variance functions of the posterior estimation
        :param x: Query inputs
        :return: Mean and variance functions
        '''
        cov_y_f = self.rbf(self.X_total, x, self.param)
        mean_outputs = np.dot(cov_y_f.T, self.beta)
        var = (self.param[0]**2 - np.diag(np.dot(np.dot(cov_y_f.T, self.inv_cov_y_y_total), cov_y_f))).reshape(-1, 1)
        xx = x.T
        mu = self.GMM(self.X_total,self.y_total,xx)
        return mean_outputs, var, mu 

    def callback(self, param, state):
        # ToDo: add something you want to know about the training process
        if state.nit % 100 == 0 or state.nit == 1:
            print('---------------------------------- iter ', state.nit, '----------------------------------')
            print('running time: ', state.execution_time)
            print('obj_cost: ', state.fun)
            print('maximum constr_violation: ', state.constr_violation)
            
