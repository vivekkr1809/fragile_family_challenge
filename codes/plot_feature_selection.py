"""
Plot the feature-selection values

Author(s) : Vivek Kumar
            vivekk@princeton.edu

            Victor Charpentier
            vc6@princeton.edu

Last Updated : 04-03-2018

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import platform

if __name__ == '__main__':
	

	feature_selection_data = np.genfromtxt('feature_selection_gpa_scores.csv',delimiter=',')
	
	fig, ax = plt.subplots(figsize=(8, 6))
	x = np.linspace(1, len(feature_selection_data) ,num=len(feature_selection_data))

	plt.grid('on')
	plt.rc('font', size=14)

	plt.plot(x,feature_selection_data,'r',linewidth = 2.)
	plt.title('Mutual Info Regression Score', fontsize = 14)
	plt.xlabel('Feature number', fontsize = 14)
	plt.ylabel('Mutual Info Score', fontsize = 14)
	plt.legend(loc='best')
	plt.savefig('feature_selection_gpa_scores.eps',bbox_inches='tight', format='eps', dpi=1000)
	plt.show()