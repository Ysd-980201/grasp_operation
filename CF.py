import numpy as np
import copy

class Obstacle_Dynamic_CF_IMP_Vol():
    def __init__(self, center = np.zeros(2), axis = np.ones(2),
        coeffs = np.ones(2), lmbda = 1.0, beta = 1.0, eta = 1.0, **kwargs):
        self.center = copy.deepcopy(center)
        self.axis = copy.deepcopy(axis)
        self.coeffs = copy.deepcopy(coeffs)
        self.lmbda = copy.deepcopy(lmbda)
        self.beta = copy.deepcopy(beta)
        self.n_dim = np.size(self.axis)
        self.eta = copy.deepcopy(eta)
        self.center_ori = self.center
        return

    def compute_cos_theta(self, x, v):
        nabla = self.compute_grad_isopot(x)
        cos_theta = np.dot(nabla, v) / np.linalg.norm(v) / np.linalg.norm(nabla)
        theta = np.nan_to_num(np.arccos(cos_theta))
        return [cos_theta, theta]

    def gen_external_force(self, x, v, Uatt, x_goal):#current state
        # print(self.center)
        # Computing the necessary quantities
        if np.linalg.norm(v) < 1e-15:
            phi = np.zeros(self.n_dim)#
        else:                    
            phi = self.fcf(x, x_goal, v, Uatt)
        return phi#array([0., 0., 0.])

    def selection_force_3d(self, fhi, repulsive_vector, v, cx):
        fr = repulsive_vector.reshape(3,1)
        rz = np.linalg.norm(fr) * np.array([[0],[0],[1]])
        alpha = np.arccos(np.dot(fr.squeeze(), rz.squeeze()) / np.linalg.norm(fr) / np.linalg.norm(rz))
        alpha = np.nan_to_num(alpha)
        r_alpha = np.cross(fr.squeeze(), rz.squeeze()) / np.linalg.norm(fr) / np.linalg.norm(rz)
        R = 1 + r_alpha * np.sin(alpha) + (r_alpha ** 2) * (1 - np.cos(alpha))
        R = R.reshape(3,1)
        R_inv = np.linalg.pinv(R)
        R_inv = R_inv.squeeze()
        gamma = 1
        lamada = gamma * np.linalg.norm(v) / cx
        x = np.cos(fhi)
        y = np.sin(fhi)
        z = 0 * x
        fs2 = np.array([[x],[y],[z]])
        fs2 = fs2.squeeze()
        select_force = lamada * R_inv * fs2
        select_force = np.array(select_force).squeeze()
        return select_force
    def selection_force_2d(self, fhi, repulsive_vector,v, cx):
        fr = repulsive_vector.reshape(2,1)
        rz = np.linalg.norm(fr) * np.array([[0],[1]])
        alpha = np.arccos(np.dot(fr.squeeze(), rz.squeeze()) / np.linalg.norm(fr) / np.linalg.norm(rz))
        alpha = np.nan_to_num(alpha)
        r_alpha = np.cross(fr.squeeze(), rz.squeeze()) / np.linalg.norm(fr) / np.linalg.norm(rz)
        R = 1 + r_alpha * np.sin(alpha) + (r_alpha ** 2) * (1 - np.cos(alpha))
        R_inv = 1 / R
        gamma = 50#50 10 70
        lamada = gamma * np.linalg.norm(v) / cx
        x = fhi#np.cos(fhi)
        y = 0 * x
        fs2 = np.array([[x],[y]])
        fs2 = fs2.squeeze()
        select_force = lamada * R_inv * fs2
        select_force = np.array(select_force).squeeze()
        return select_force
    
    def compute_grad_isopot(self, x):
        grad = np.zeros(self.n_dim)#
        for _i in range(self.n_dim):
            grad[_i] = 2.0 * self.coeffs[_i] * \
                (x[_i] - self.center[_i]) ** (2.0 * self.coeffs[_i] - 1.0) / \
                self.axis[_i] ** (2.0 * self.coeffs[_i])
        return grad

    def compute_nabla_dot_prod(self, x, v):
        nabla = np.zeros(self.n_dim)
        for _i in range(self.n_dim):
            nabla[_i] = 2.0 * self.coeffs[_i] * (2.0 * self.coeffs[_i] - 1.0) * \
                (x[_i] - self.center[_i]) ** (2.0 * self.coeffs[_i] - 2.0) * \
                v[_i] / self.axis[_i] ** (2.0 * self.coeffs[_i])
        return nabla

    def compute_nabla_norm(self, x):
        norm_nabla = np.linalg.norm(self.compute_grad_isopot(x))
        nabla = np.zeros(self.n_dim)
        for _i in range(self.n_dim):
            nabla[_i] = 4.0 * self.coeffs[_i] * self.coeffs[_i] * \
                (2.0 * self.coeffs[_i] - 1.0) * \
                (x[_i] - self.center[_i]) ** (4.0 * self.coeffs[_i] - 3.0) / \
                self.axis[_i] ** (4.0 * self.coeffs[_i])
        return nabla / norm_nabla

    def compute_potential(self, x, v):
        [cos_theta, theta] = self.compute_cos_theta(x, v)
        isopot = self.compute_isopotential(x)
        potential = self.lmbda * (- cos_theta) ** self.beta * \
            np.linalg.norm(v) / isopot 
        
        potential[theta < np.pi / 2.0] = 0.0
        return potential

    def compute_isopotential(self, x):
        K = 0.
        for i in range(self.n_dim):
            K += ((x[i] - self.center[i]) / self.axis[i]) ** (2 * self.coeffs[i])
        K -= 1
        return K#   
       
    def fcf(self, x, x_goal, v, uatt):        
        d = np.zeros(self.n_dim)
        for _i in range(self.n_dim):
            d[_i] = 2.0 * self.coeffs[_i] * \
                (self.center[_i] - x[_i]) ** (2.0 * self.coeffs[_i] - 1.0) / \
                self.axis[_i] ** (2.0 * self.coeffs[_i])
        if self.n_dim == 3:
            n = np.array([d[0], d[1], d[2]])
            cx = self.compute_isopotential(x)
            # print(cx)
            """========== """
            # if cos_theta < 0.85:#0.85 #np.linalg.norm(self.center - x) 
            if np.linalg.norm(self.center - x)  < 0.2:
                fhi = 0
                select_force_list = np.array(np.zeros(3))
                select_force_list = select_force_list.reshape(1,3)
                for j in range(100):#60
                    if j < 50:#30
                        fhi = fhi + np.random.rand(1) * np.pi
                    else:
                        fhi = fhi - np.random.rand(1) * np.pi
                    select_force = self.selection_force_3d(fhi, n, v,cx)
                    select_force = np.array([select_force])
                    select_force_list = np.append(select_force_list, select_force, axis=0)
                select_force_list = select_force_list[1:,:]
                sita_list = np.array(np.zeros(1))
                for i in range(select_force_list.shape[0]):
                    x_g = x - x_goal
                    # x_g = uatt - x
                    cos_sita = np.dot(x_g, select_force_list[i,:]) / np.linalg.norm(select_force_list[i,:]) / np.linalg.norm(x_g)
                    sita = np.nan_to_num(np.arccos(cos_sita))
                    sita_list = np.append(sita_list, np.array([sita]), axis=0)
                sita_list = sita_list[1:,]
                min_idx = np.argmin(sita_list)
                select_force_list = select_force_list[min_idx, :]
            """========== """
            d_ox = self.center - x
            projection = np.cross(d_ox, n)
            normal = projection / np.linalg.norm(projection)
            # if cos_theta < 0.85:#0.85 0.5
            if np.linalg.norm(self.center - x)  < 0.2:
                normal = select_force_list / np.linalg.norm(select_force_list)
            b = x_goal - x
            c = np.cross(normal, b)#normal * b
            B = np.cross(c, v)
            d_star = v / np.linalg.norm(v)
            cx = self.compute_isopotential(x)
            k_cf = 100.0
            fcf = np.cross(k_cf / cx * d_star, B)
        elif self.n_dim == 2:
            n = np.array([d[0], d[1]])
            cx = self.compute_isopotential(x)
            """==============="""
            if uatt is not None:
                dis = self.compute_isopotential(uatt)
                dis_max = 3.0#
            else:
                dis_max = 0.5
            fhi = 0
            select_force_list = np.array(np.zeros(2))
            """==============="""
            d_ox = self.center - x
            cossita = np.dot(d_ox, n) / np.linalg.norm(n) / np.linalg.norm(d_ox)
            projection = d_ox * cossita # 
            normal = projection / np.linalg.norm(projection)
            if dis < dis_max: #cos_theta > np.pi / 4  2.0 < theta < np.pi / 1 np.pi / 2 < theta <= np.pi:
                b = x_goal - x  # (uatt - x) / np.linalg.norm(uatt - x) #
                c = normal * b
                B = c * v
                d_star = v / np.linalg.norm(v)/(1+np.exp(-v))
                k_cf = 100.0 #100
                fcf = k_cf / cx * d_star * B /(1+np.exp(-v))/np.linalg.norm(v)# + 30 * select_force_list #- 30 * v           
                select_force_list = select_force_list.reshape(1,2)
                for j in range(2):
                    if j < 1:
                        fhi = 1
                    else:
                        fhi = -1
                    select_force = self.selection_force_2d(fhi, fcf, v, cx)
                    select_force = np.array([select_force])
                    select_force_list = np.append(select_force_list, select_force, axis=0)
                select_force_list = select_force_list[1:,:]
                sita_list = np.array(np.zeros(1))
                for i in range(select_force_list.shape[0]):
                    x_g = x_goal - x
                    cos_sita = np.dot(x_g, select_force_list[i,:]) / np.linalg.norm(x_g) / np.linalg.norm(select_force_list[i,:])
                    sita = np.nan_to_num(np.arccos(cos_sita))
                    sita_list = np.append(sita_list, np.array([sita]), axis=0)
                sita_list = sita_list[1:,]
                min_idx = np.argmin(sita_list)
                select_force_list = select_force_list[min_idx, :]
                fcf = fcf +  select_force_list
            else:
                b = x_goal - x 
                c = normal * b
                B = c * v
                d_star = v / np.linalg.norm(v)
                k_cf = 100.0 #
                fcf = k_cf / cx * d_star * B
        else:
            fcf = 0.0

        return fcf
