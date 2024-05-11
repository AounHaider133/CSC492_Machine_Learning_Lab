import numpy as np
import matplotlib.pyplot as plt

def estimate_coeff(x,y):
 
 mean_x = np.mean(x)
 mean_y = np.mean(y)
 
 SSxy = np.sum(x*y) - np.size(x)*mean_x*mean_y
 SSxx = np.sum(x*x) - np.size(x)*mean_x*mean_x
 
 b1 = SSxy/SSxx
 b0 = mean_y - b1*mean_x 
 
 return (b0, b1)
 
def plot_regression_line(x,y,b):
 plt.scatter(x,y,color = "m", marker = 's', s =60)
 
 predicted_y = b[0]+b[1]*x
 
 plt.plot(x,predicted_y,color="y")
 plt.title('Linear Regression')
 plt.xlabel('size(ft)')
 plt.ylabel('price')
 
 plt.show()

def main():
 
 x = np.array([0,1,2,3,4,5,6,7,8,9])
 y = np.array([1,3,2,5,7,8,8,9,10,12])
 
 #Get b0 & b1 as tuple
 b = estimate_coeff(x,y)
 
 print("Estimated co-efficients:\nb0:{}\nb1:{}".format(b[0],b[1]))
 
 #plot regression line
 plot_regression_line(x,y,b)

if __name__ == '__main__':
 main()
