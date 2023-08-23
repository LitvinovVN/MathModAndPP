import matplotlib.pyplot as plt

x = []
T = []

myfile = open("out.txt", "r")

for line in myfile:
    data = line.split()
    x.append(float(data[0]))
    T.append(float(data[1]))

myfile.close()

plt.plot(x, T)
plt.show()