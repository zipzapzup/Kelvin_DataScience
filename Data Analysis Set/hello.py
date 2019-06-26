print("Hello")
import sys

print(sys.version)


#Tuples - Immutable
ratings = (10,2,9,19,20)


#Lists - Mutable
list =[1,3,5,6,7]
# Method that can be used
# Append
# Extend
# Split



#Sets - Does not contain duplicate
#sets= set(1,2,3,4,5)

# Method that can be used
# "A" in sets
# .remove() to remove something in the sets
# sets3 = set1 & set2 This method look at the union between the 2 sets
# .add() Also add a new element in the sets
# .difference can also be used to find the difference between 2 sets. Examples: sets1.difference(sets2)
# .intersection - used to find the mid point that intersects. sets1.intersection(sets2)
# .union is used to determine all of the points.
#


#Dictionary - Dictionary is a set of key and values pair. Creating dictionary is used by using curly bracketsself.

Dict = {"key1":1, "key2":"2"}

# Accessing dictionary values can be accessed via its keys.
Dict["key1"]
# Dictionary Methods:
# del(Dict["keys1"])
# Append or add more Dictionary:
# Dict["keys4"] = 5
# Dict.keys()
# Dict.values()



#Loops
# 2 Different type of loops, FOR LOOP and WHILE Loops
# FOR
# for i in test:
#     print (i)
# Using enumerate, you will get the index and then the values next
# for i,b in enumerate(lists):
#     print(i,b)

# WHILE Loops
# while(condition):
#     print("Keep on going")

with open("/Users/kelvinjumino/Documents/Project/Python Project/test.txt","r") as FILE1:
    File = FILE1.read()
    print(File)



# pandas library

# >>> df = pd.DataFrame({'a':[11,21,31],'b':[21,22,23]})
# >>> df.head(3)
#    a   b
# 0  11  21
# 1  21  22
# 2  31  23
# >>> df['a']
# 0    11
# 1    21
# 2    31
# Name: a, dtype: int64
# >>> df['b']
# 0    21
# 1    22
# 2    23
# Name: b, dtype: int64
# >>>



# Vector:
# u = [0,1] The 1 in the right hand side mean it goes up by 1 vertically
# u = [1,0] The 1 in the left mean it goes horizontally by 1


# Pandas is used to display data DataFrame
# Numpy is used to calculate numberical functions, such as Vector
# Matplotlib is used to plot graphs

# Numpy:
# np.linspace(start,end, number of frequency)
# x is usually the array or the x axis
# y is the y axis
# Before you show the graph, you usually need to import the Matplotlib
# import matplotlib.pyplot as plt
# x=np.linspace(0,2*np.pi,100)
# y = np.sin(x) + 2
#
# plt.plot(x,y)
# plt.show()




# 2 Functions for plotting vector

import time
import sys
import numpy as np

import matplotlib.pyplot as plt

def Plotvec1(u,z,v):
    #this function is used in code
    ax = plt.axes()
    ax.arrow(0, 0, *u, head_width=0.05,color ='r', head_length=0.1)
    plt.text(*(u+0.1), 'u')

    ax.arrow(0, 0, *v, head_width=0.05,color ='b', head_length=0.1)
    plt.text(*(v+0.1), 'v')
    ax.arrow(0, 0, *z, head_width=0.05, head_length=0.1)
    plt.text(*(z+0.1), 'z')
    plt.ylim(-2,2)
    plt.xlim(-2,2)



def Plotvec2(a,b):
    #this function is used in code
    ax = plt.axes()
    ax.arrow(0, 0, *a, head_width=0.05,color ='r', head_length=0.1)
    plt.text(*(a+0.1), 'a')
    ax.arrow(0, 0, *b, head_width=0.05,color ='b', head_length=0.1)
    plt.text(*(b+0.1), 'b')

    plt.ylim(-2,2)
    plt.xlim(-2,2)
