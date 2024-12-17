import numpy as np

class GMM:
    def __init__(self, num_components=2, max_iters=200, tol=1e-6):
        #GMMクラスの初期化
        self.num_components = num_components #混合数
        self.max_iters = max_iters #最大反復回数
        self.tlo = tol #収束条件
        
    
    def initialize_parameters(self, data):
        
        N, D = data.shape
        indices = np.random.choice(N, self.num_components, replace = False)
        
        #平均ベクトルをランダムに初期化
        self.means = data[indices]
        
        #単位行列で初期化
        self.covariances = np.array([np.eye(D) for _ in range(self.num_components)])
        
        #重みを均等に初期化
        self.weights = np.ones(self.num_components) / self.num_components
        
    #Eステップ 
    def e_step(self, data):
       
       #負担率を計算する
        N, D = data.shape
        responsibilities = np.zeros((N, self.num_components))
        
        for m in range(self.num_components):
            pdf = self.multivariate_gaussian(data, self.means[m], self.covariances[m])
            responsibilities[:, m] = self.weights[m] * pdf
        
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities   
    
    #Mステップ
    def m_step(self, data, responsibilities):
        
        N, D = data.shape
        for m in range(self.num_components):
            resp = responsibilities[:, m]
            total_resp = resp.sum()
            self.means[m] = np.sum(resp[:, np.newaxis] * data, axis=0) / total_resp
            diff = data - self.means[m]
            self.covariances[m] = (resp[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', diff, diff)).sum(axis=0) / total_resp
            self.weights[m] = total_resp / N
            
    #対数尤度を計算する関数
    def log_likelihood(self, data):
        """
        対数尤度を計算する
        """
        N, D = data.shape
        log_likelihood = 0
        for n in range(N):
            likelihood = 0
            for m in range(self.num_components):
                pdf = self.multivariate_gaussian(data[n], self.means[m], self.covariances[m])
                likelihood += self.weights[m] * pdf
            log_likelihood += np.log(likelihood)
        return log_likelihood
    
    
    def fit(self, data):
       
        self.initialize_parameters(data)
        log_likelihoods = []
        
        for iteration in range(self.max_iters):
            responsibilities = self.e_step(data)
            self.m_step(data, responsibilities)
            log_likelihood = self.log_likelihood(data)
            log_likelihoods.append(log_likelihood)
            
            if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                break
        
        return log_likelihoods
    
    def multivariate_gaussian(self, x, mean, covariance):
        
        D = x.shape[0]
        det = np.linalg.det(covariance)
        inv = np.linalg.inv(covariance)
        norm_const = 1.0 / (np.sqrt((2 * np.pi) ** D * det))
        x_centered = x - mean
        return norm_const * np.exp(-0.5 * x_centered @ inv @ x_centered)
    
    
    
    
   
        