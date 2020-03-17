import numpy as np
import pylab as pl
from numpy import arctan as atan
from numpy import pi as pi
from numpy import sqrt as sqrt
from numpy import log as log
from numpy import log10 as log10
from scipy.optimize import curve_fit
import cmath as cm


RCS, TOT, DEC, DECICU, t = np.loadtxt("DatiCOVID10.txt", unpack=True)
dRCS=np.sqrt(RCS)
dTOT=np.sqrt(TOT)
dDEC=np.sqrt(DEC)
dDECICU=np.sqrt(DECICU)

##Modello Logistico
print('MODELLO LOGISTICO')
#Definisco il modello
def model(x, L,t0,k):
    return L/(1+np.exp(-(x-t0)/k))

#Determino i migliori valori di M e q
parin=(4000,23,4)
pars, covm = curve_fit(model, t, DEC,parin, sigma=dDEC, absolute_sigma=False)
print('L = %.5f +- %.5f ' % (pars[0], np.sqrt(covm.diagonal()[0])))
print('t0 = %.5f +- %.5f ' % (pars[1], np.sqrt(covm.diagonal()[1])))
print('Tau = %.5f +- %.5f ' % (pars[2], np.sqrt(covm.diagonal()[2])))


#Chi quadro
#print(covm)
chisq = sum(((DEC - model(t, *pars))/dDEC)**2.)
ndof = len(DEC) - 3
print('chi quadro = %.3f (%d dof)' % (chisq, ndof))

##Modello Esponenziale
print('MODELLO ESPONENZIALE')
#Definisco il modello
def model1(x, L, t0):
    return L*np.exp(x/t0)

#Determino i migliori valori di M e q
pars1, covm1 = curve_fit(model1, t, DEC, sigma=dDEC, absolute_sigma=False)
print('L = %.5f +- %.5f ' % (pars1[0], np.sqrt(covm1.diagonal()[0])))
print('t0 = %.5f +- %.5f ' % (pars1[1], np.sqrt(covm1.diagonal()[1])))

#Chi quadro
chisq1 = sum(((DEC - model1(t, *pars1))/dDEC)**2.)
ndof1 = len(DEC) - 2
print('chi quadro = %.3f (%d dof)' % (chisq1, ndof1))

##Grafico principale Logistico
fig1 = pl.figure(1)
frame1=fig1.add_axes((.1,.3,.8,.6))
frame1.set_title('Number of deaths VS Logistic model',fontsize=20)
x=np.linspace(min(t),20, 5000)
y=model(x, *pars)
pl.plot(x,y, color='red', alpha=0.5)
pl.errorbar(x=t, y=DEC, yerr=dDEC, fmt='.', color='black')
pl.ylabel('$Number$ [Unity]',fontsize=15)
pl.xlabel('$Time$ [Day]',fontsize=15)
pl.text(1,1500, 'L = %.3f +- %.3f ' % (pars[0], np.sqrt(covm.diagonal()[0])), fontsize=15)
pl.text(1,1100, 't0 = %.3f +- %.3f ' % (pars[1], np.sqrt(covm.diagonal()[1])), fontsize=15)
pl.text(1,800, 'Tau = %.3f +- %.3f ' % (pars[2], np.sqrt(covm.diagonal()[2])), fontsize=15)
pl.text(1,600, 'Chi-square = %.3f (%d dof)' % (chisq, ndof), fontsize=15)
pl.yscale('log')

##Residui
ff=(DEC-model(t, *pars))
frame2=fig1.add_axes((.1,.1,.8,.2))
pl.plot(x, 0*x, color='red', linestyle='--', alpha=0.5)
pl.errorbar(t, ff,yerr=dDEC, fmt='.', color='black')
#pl.yticks(range(-150, 150))
frame2.set_ylabel('Residuals logistic model')
pl.xlabel('$Time$ $from$ $24/02/2020$ [Day]',fontsize=15)


##Grafico principale Esponenziale
fig2 = pl.figure(2)
frame1=fig2.add_axes((.1,.3,.8,.6))
frame1.set_title('Number of deaths VS Exponential model',fontsize=20)
x1=np.linspace(min(t),20, 5000)
y1=model1(x1, *pars1)
pl.plot(x1,y1, color='red', alpha=0.5)
pl.errorbar(x=t, y=DEC, yerr=dDEC, fmt='.', color='black')
pl.ylabel('$Number$ [Unity]',fontsize=15)
pl.text(1,1500, 'L = %.3f +- %.3f ' % (pars1[0], np.sqrt(covm1.diagonal()[0])), fontsize=15)
pl.text(1,1100, 'Tau = %.3f +- %.3f ' % (pars1[1], np.sqrt(covm1.diagonal()[1])), fontsize=15)
pl.text(1,800, 'Chi-square = %.3f (%d dof)' % (chisq1, ndof1), fontsize=15)
pl.yscale('log')

##Residui
ff1=(DEC-model1(t, *pars1))
frame2=fig2.add_axes((.1,.1,.8,.2))
pl.plot(x1, 0*x1, color='red', linestyle='--', alpha=0.5)
pl.errorbar(t, ff1,yerr=dDEC, fmt='.', color='black')
frame2.set_ylabel('Residuals logistic model')
pl.xlabel('$Time$ $from$ $24/02/2020$ [Day]',fontsize=15)
#pl.yticks(range(-150, 150))



pl.show()