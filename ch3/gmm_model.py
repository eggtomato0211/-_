import numpy as np
import matplotlib.pyplot as plt

class GMM:
    def __init__(self, num_components=2, max_iters=200, tol=1e-6):
        self.num_components = num_components
        self.max_iters = max_iters
        self.tol = tol

    def initialize_parameters(self, data, custom_means=None):
        N, D = data.shape
        # データの最小値と最大値を使って、範囲内にランダムな点を生成
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        # 初期平均ベクトルを手動設定またはランダム初期化
        if custom_means is None:
            self.means = np.random.uniform(low=min_vals, high=max_vals, size=(self.num_components, D))
        else:
            self.means = custom_means   
        
        # --- 対角共分散行列を初期化（各成分の分散をランダムに設定）---
        data_cov = np.cov(data, rowvar=False)  # データの共分散行列を求める
        init_variances = np.diag(data_cov)  # 対角成分を取得（データの各成分の分散）
        self.covariances = np.array([
            np.diag(np.random.uniform(0.5 * init_variances, 1.5 * init_variances)) 
            for _ in range(self.num_components)
        ])
        
         # --- ランダムに重みを生成し、正規化する ---
        random_weights = np.random.rand(self.num_components)
        self.weights = random_weights / np.sum(random_weights)  # 和が1になるように正規化
        

        

    def e_step(self, data):
        N, D = data.shape
        responsibilities = np.zeros((N, self.num_components))
        
        for m in range(self.num_components):
            pdf = self.multivariate_gaussian(data, self.means[m], self.covariances[m])
            responsibilities[:, m] = self.weights[m] * pdf
        
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, data, responsibilities):
        N, D = data.shape
        for m in range(self.num_components):
            resp = responsibilities[:, m]
            total_resp = resp.sum()
            self.means[m] = np.sum(resp[:, np.newaxis] * data, axis=0) / total_resp
            diff = data - self.means[m]
            self.covariances[m] = (resp[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', diff, diff)).sum(axis=0) / total_resp
            self.weights[m] = total_resp / N

    def log_likelihood(self, data):
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
        
        if x_centered.ndim == 1:  
            exponent = -0.5 * x_centered.T @ inv @ x_centered
        else:
            exponent = -0.5 * np.einsum('ij,jk,ik->i', x_centered, inv, x_centered)
        
        return norm_const * np.exp(exponent)

    def plot_log_likelihood(self, log_likelihoods, num_components, condition_index = None):
        plt.plot(log_likelihoods)
        plt.title(f'Log-Likelihood Convergence (num_components={num_components})')
        plt.xlabel('Iterations')
        plt.ylabel('Log-Likelihood')
        if condition_index is not None:
            filename = f'log_likelihood_{num_components}_condition_{condition_index + 1}.png'
        else:
            filename = f'log_likelihood_{num_components}.png'
        plt.savefig(filename)  # 動的なファイル名で保存
        plt.close()

    def plot_means(self, num_components):
        for m in range(self.num_components):
            plt.scatter(self.means[m][0], self.means[m][1], label=f'Mean {m+1}')
        plt.title(f'Estimated Means (num_components={num_components})')
        plt.xlabel('Temperature')
        plt.ylabel('Humidity')
        plt.legend()
        plt.savefig(f'means_{num_components}.png')  # PNGファイルとして保存
        plt.close()  # プロットを閉じてメモリを解放
