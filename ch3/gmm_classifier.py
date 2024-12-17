#必要ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

#データの可視化
def load_binary_data(filename, num_sumples=300, dimensions=2):
    #バイナリーデータを読み込む
    data = np.fromfile(filename, dtype= np.float32)
    return data.reshape((num_sumples, dimensions))


#データの読み込み
class1_data = load_binary_data('class1.dat')
class2_data = load_binary_data('class2.dat')

# 散布図を表示
plt.figure(figsize=(8, 6))
plt.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1 (Kobe)', color='blue', alpha=0.6)
plt.scatter(class2_data[:, 0], class2_data[:, 1], label='Class 2 (Naha)', color='orange', alpha=0.6)
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Scatter Plot of Class 1 (Kobe) and Class 2 (Naha)')
plt.legend()
plt.show()