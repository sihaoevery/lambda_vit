from tkinter.tix import X_REGION
import numpy as np
import torch
import torch.fft as fft
import math
def Fourier_trans(x):
    '''
        x: B,C,H,W
        get the fft frequency of input x
    '''
    x_freq = fft.fftn(x, dim=(-2, -1)) # data type is complex number
    x_freq = fft.fftshift(x_freq, dim=(-2, -1)) # the center is of 0 frequency

    return x_freq

s=set()
all_distance = []
for x in range(14):
    for y in range(14):
        freq = math.sqrt(((x-7)/7)**2+((y-7)/7)**2)
        all_distance.append([freq,x,y])
        s.add(freq)
        if freq == 1.:
            print(x,y)

dist = np.array(all_distance)
# print(sorted(list(s)))

# possible distance, 1 means 1 pi, 0 meas 0 pi
possible_dist = [0.0, 0.14285714285714285, 0.20203050891044214, 0.2857142857142857, 0.3194382824999699, \
        0.40406101782088427, 0.42857142857142855, 0.4517539514526256, 0.5150787536377127, \
        0.5714285714285714, 0.5890150893739515, 0.6060915267313264, 0.6388765649999398, \
        0.7142857142857142, 0.7142857142857143, 0.7284313590846836, 0.769309258162072, \
        0.8081220356417685, 0.8329931278350429, 0.8571428571428571, 0.8689660757568884, \
        0.9035079029052512, 0.9147320339189784, 0.9583148474999098, 1.0, 1.0101525445522108, \
        1.0301575072754254, 1.0400156984686455, 1.0879675865519869, 1.1157499537009505, \
        1.1517511068997928, 1.2121830534626528, 1.228903609577518, 1.3170777796132696, 1.4142135623730951]

# hard-code, 0, 0.1 pi, ..., pi
idx = [0, 0.14285714285714285, 0.20203050891044214, 0.3194382824999699,0.40406101782088427,0.5150787536377127, \
    0.6060915267313264, 0.7142857142857142, 0.8081220356417685, 0.9035079029052512, 1.]

# for i in idx:
#     res = dist[dist[:,0]==i]
#     print(f'idx:{i},len:{len(res)},{res}')

def cal_distribution(x):
    x_freq = Fourier_trans(torch.tensor(x))
    B,C,H,W = x_freq.shape
    stats=[]
    for i in idx:
        res = dist[dist[:,0]==i]
        sum = 0
        for j in res[:,1:].astype(np.int32): # iterate pixels
            tmp = x_freq[:,:, j[0],j[1]]
            tmp = torch.view_as_real(tmp)
            tmp = tmp**2
            tmp = torch.sum(tmp,dim=2)
            tmp = torch.sqrt(tmp)
            sum += tmp.sum()
        stats.append(sum/(B*C*len(res)))
    return stats

if __name__ == '__main__':
    x = np.random.randn(128,512,14,14)
    testresult =cal_distribution(x) 
    print(testresult.numpy())