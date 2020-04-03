import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from scipy.integrate import solve_ivp
import os
from scipy.optimize import minimize

########## SETTINGS #############
county = "example"
county_pop = 460001
predict_range = 60
complexity = 0.0000001  # original is 0.00000001
#################################

# Susceptible equation
def fa(N, a, b, beta):
    fa = -beta*a*b
    return fa

# Infected equation
def fb(N, a, b, beta, gamma):
    fb = beta*a*b - gamma*b
    return fb

# Recovered/deceased equation
def fc(N, b, gamma):
    fc = gamma*b
    return fc

# Runge-Kutta method of 4rth order for 3 dimensions (susceptible a, infected b and recovered r)
def rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs):
    a1 = fa(N, a, b, beta)*hs
    b1 = fb(N, a, b, beta, gamma)*hs
    c1 = fc(N, b, gamma)*hs
    ak = a + a1*0.5
    bk = b + b1*0.5
    ck = c + c1*0.5
    a2 = fa(N, ak, bk, beta)*hs
    b2 = fb(N, ak, bk, beta, gamma)*hs
    c2 = fc(N, bk, gamma)*hs
    ak = a + a2*0.5
    bk = b + b2*0.5
    ck = c + c2*0.5
    a3 = fa(N, ak, bk, beta)*hs
    b3 = fb(N, ak, bk, beta, gamma)*hs
    c3 = fc(N, bk, gamma)*hs
    ak = a + a3
    bk = b + b3
    ck = c + c3
    a4 = fa(N, ak, bk, beta)*hs
    b4 = fb(N, ak, bk, beta, gamma)*hs
    c4 = fc(N, bk, gamma)*hs
    a = a + (a1 + 2*(a2 + a3) + a4)/6
    b = b + (b1 + 2*(b2 + b3) + b4)/6
    c = c + (c1 + 2*(c2 + c3) + c4)/6
    return a, b, c


def SIR(N, b0, beta, gamma, hs):
    """
    N = total number of population
    beta = transition rate S->I
    gamma = transition rate I->R
    k =  denotes the constant degree distribution of the network (average value for networks in which
    the probability of finding a node with a different connectivity decays exponentially fast
    hs = jump step of the numerical integration
    """

    # Initial condition
    a = float(N - 1) / N - b0
    b = float(1) / N + b0
    c = 0.

    sus, inf, rec = [], [], []
    for i in range(10000):  # Run for a certain number of time-steps
        sus.append(a)
        inf.append(b)
        rec.append(c)
        a, b, c = rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs)

    return sus, inf, rec

def show_sample_sir_model():
    # Parameters of the model
    N = 7800*(10**6)
    b0 = 0
    beta = 0.7
    gamma = 0.2
    hs = 0.1

    sus, inf, rec = SIR(N, b0, beta, gamma, hs)

    f = plt.figure(figsize=(8,5))
    plt.plot(sus, 'b.', label='susceptible')
    plt.plot(inf, 'r.', label='infected')
    plt.plot(rec, 'c.', label='recovered/deceased')
    plt.title("SIR model")
    plt.xlabel("time", fontsize=10)
    plt.ylabel("Fraction of population", fontsize=10)
    plt.legend(loc='best')
    plt.xlim(0, 1000)
    #plt.savefig('SIR_example.png')
    plt.show()


def show_custom_sir_model(pop, beta, gamma):
    # Parameters of the model
    N = float(pop)
    b0 = 0
    #beta = 0.7
    #beta = beta/100.0
    #gamma = gamma/100.0
    #gamma = 0.2
    hs = 0.1

    sus, inf, rec = SIR(N, b0, beta, gamma, hs)

    f = plt.figure(figsize=(8,5))
    plt.plot(sus, 'b.', label='susceptible')
    plt.plot(inf, 'r.', label='infected')
    plt.plot(rec, 'c.', label='recovered/deceased')
    plt.title("Custom SIR model (prbably bad)")
    plt.xlabel("time", fontsize=10)
    plt.ylabel("Fraction of population", fontsize=10)
    plt.legend(loc='best')
    plt.xlim(0,150)
    #plt.savefig('SIR_example.png')
    plt.show()


# The SIR model differential equations.
def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def fit_odeint(xdata, beta, gamma):
    return integrate.odeint(sir_model, (sus0, inf0, rec0), xdata, args=(N, beta, gamma))[:,1]

def get_per_day_from_total(arr):
    newc = []
    newc.append(arr[0])
    for i in range(1, len(arr)):
        newc.append(arr[i]-arr[i-1])
    return newc

#################################
# show_sample_sir_model()
#################################
df = pd.read_csv('data' +os.path.sep + county+'.csv')
ydata = df['total_cum'].values.tolist()
xdata = list(range(0, len(ydata)))
dates = df['date'].values.tolist()
recovered = df['total_recovered'].values.tolist()
deaths = df['total_deaths'].values.tolist()

N = float(county_pop)
inf0 = ydata[0]
sus0 = N - inf0
rec0 = 0.0

popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
fitted = fit_odeint(xdata, *popt)

plt.plot(xdata, ydata, 'o')
plt.plot(xdata, fitted)
plt.title("Fit of SIR model to " + county)
plt.ylabel("#Persons Infected")
plt.xlabel("Days Since First Case - " + dates[0])
plt.show()
print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
#show_custom_sir_model(N, popt[0], popt[1])
###########################################################

def extend_index(values, new_size):
    for i in range(len(values), len(values)+new_size):
        values = np.append(values, i)
    return values


def loss(point, data, recovered, s_0, i_0, r_0):
    size = len(data)
    beta, gamma = point
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
    solution = solve_ivp(SIR, [0, size], [s_0,i_0,r_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] - data)**2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered)**2))
    alpha = 0.1
    return alpha * l1 + (1 - alpha) * l2



def predict(beta, gamma, data, recovered, death, s_0, i_0, r_0):
    new_index = extend_index(xdata, predict_range)
    size = len(new_index)

    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

    extended_actual = np.concatenate((data, [None] * (size - len(data))))
    extended_recovered = np.concatenate((recovered, [None] * (size - len(recovered))))
    extended_death = np.concatenate((death, [None] * (size - len(death))))
    return new_index, extended_actual, extended_recovered, extended_death, solve_ivp(SIR, [0, size], [s_0, i_0, r_0], t_eval=np.arange(0, size, 1))


optimal = minimize(loss, [0.001, 0.001], args=(ydata, recovered, sus0, inf0, rec0), method='L-BFGS-B', bounds=[(complexity, 0.4), (complexity, 0.4)])
print(optimal)
beta, gamma = optimal.x
new_index, extended_actual, extended_recovered, extended_death, prediction = predict(beta, gamma, ydata, recovered, deaths, sus0, inf0, rec0)
df = pd.DataFrame(
    {'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death, 'Susceptible': prediction.y[0], 'Infected': prediction.y[1],
     'Recovered': prediction.y[2]}, index=new_index)
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title(county)
df.plot(ax=ax)
print(f"country={county}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta / gamma):.8f}")
plt.title("SIR model prediction of " + county)
plt.ylabel("#Persons Infected")
plt.xlabel("Days Since First Case - " + dates[0])
plt.show()
