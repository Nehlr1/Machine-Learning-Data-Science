"""""""""
First NumPy coding

"""""""""

import numpy as np

Num_Array = np.array([[1,2,3],[4,5,6]])

NP1 = np.array([[1,3],[4,5]])

NP2 = np.array([[3,4],[5,7]])

MNP = NP1@NP2

MNP_3 = np.dot(NP1, NP2)

MNP_2 = NP1*NP2

MNP_4 = np.multiply(NP1, NP2)

print(MNP_3)

SUM1 = NP1+NP2

SUB1 = NP1-NP2

SUB2 = np.subtract(NP1, NP2)

Ele_Sum = np.sum(NP1)

Broad_Num = NP1 + 3

NP3 = np.array([[3,4]])

NP1+NP3

D = np.divide([15, 45, 56], 4)

D1 = np.floor_divide([15, 45, 56], 4)

np.math.sqrt(56)

random_Normal_Distribution = np.random.standard_normal((3,4))

random_Uniform_Distribution = np.random.uniform(1, 15, (3,4))

#Generate ramdom Integer Number

random_Integer = np.random.randint(1, 100, (3,5))

#Generate ramdom Float Number

random_Float = np.random.rand(5, 5)

filter_Ar = np.logical_and(random_Integer > 30, random_Integer < 60)

F_Random_Ar = random_Integer[filter_Ar]

Data_N = np.array([1,3,5,7,9,10])

Mean_N = np.mean(Data_N)

Median_N = np.median(Data_N)

Std_N = np.std(Data_N)

Variance_N = np.var(Data_N)

Variance_Nump = np.var(Num_Array, axis=1)

Variance_Nump_2 = np.var(Num_Array, axis=0)

