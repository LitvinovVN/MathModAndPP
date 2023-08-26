import matplotlib.pyplot as plt

i_list = []
x_list = []
T_list = []
kratn = 10000

myfile = open("out.txt", "r")

for line in myfile:
    data = line.split()
    i = float(data[0])
    x = float(data[1])
    T = float(data[2])
    if i % kratn == 0:
        i_list.append(i)
        x_list.append(x)
        T_list.append(T)

myfile.close()

plt.plot(x_list, T_list)
plt.show()