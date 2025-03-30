import numpy as np
from vmdpy import VMD
import pandas as pd
import matplotlib.pyplot as plt
import pdb

def vmd_decomposition(data, alpha, tau, K, DC, init, tol):
    """
    对输入的时间序列数据进行VMD分解
    :param data: 输入的时间序列数据
    :param alpha: 惩罚因子
    :param tau: 噪声容忍度
    :param K: 分解的模态数
    :param DC: 是否包含直流分量
    :param init: 初始化方式
    :param tol: 收敛容忍度
    :return: 分解后的模态
    """
    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    return u

def main():
    file_path = './data/DWV/DWV.csv'
    data = pd.read_csv(file_path)
    dates = data.iloc[:, 0]  # 获取前365天的日期
    data = data['Returning.Visits']
    # data = data[:365]

    # pdb.set_trace()
    alpha = 200  # 惩罚因子
    tau = 0.  # 噪声容忍度
    K = 5  # 分解的模态数
    DC = 0  # 不包含直流分量
    init = 1  # 初始化方式
    tol = 1e-7  # 收敛容忍度

    # 进行VMD分解
    u = vmd_decomposition(data, alpha, tau, K, DC, init, tol)

    # 将分解结果写入新文件
    df = pd.DataFrame(u.T)
    output_path = './data/DWV/vmd_decomposition_results.csv'
    df.to_csv(output_path, index=False)
    print("分解结果已写入 vmd_decomposition_results.csv 文件。")

    plt.figure(figsize=(10, 8))
    plt.subplot(K + 1, 1, 1)
    plt.plot(dates, data, label='Original Signal')  # 使用日期作为X轴
    plt.title('Original Signal')
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    
    for i in range(K):
        plt.subplot(K + 1, 1, i + 2)
        plt.plot(dates[1:], u[i], label=f'Mode {i + 1}')  # 使用日期作为X轴
        plt.title(f'Mode {i + 1}')
        plt.legend()
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=5))  # 设置X轴刻度，使其不那么密集

    plt.tight_layout()
    
    plt.savefig('./data/DWV/vmd_decomposition_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()