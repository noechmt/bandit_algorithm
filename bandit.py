from random import randint, random
from math import sqrt, log
from time import time
import matplotlib.pyplot as plt

number_of_try = 5
K = 2
N = 100000
cumulative_reward = [0 for i in range(10*number_of_try)]
duree = [0 for i in range(10)]
p = 0
list_display = [x*N/10 for x in range(1, 11)]

def pull(arm):
    arms = [0.4, 0.8]
    return arms[arm]

def mean_cumulative_reward(test):
    average = [0 for i in range(10)]
    for i in range(10):
        average[i] = sum(test[i+x*10] for x in range(0,number_of_try))/number_of_try
    return average

def plot_cumulative_reward(name, cumulative_reward, list_display):
    plt.figure("reward")
    plt.ylabel("Cumulative reward")
    plt.xlabel("Number of pulls")
    match name:
        case "random":
            plt.plot(list_display, cumulative_reward, "-r^", label='random')
        case "eps_greedy":
            plt.plot(list_display, cumulative_reward, "-go", label='eps_greedy')
        case "eps_greedy_dec":
            plt.plot(list_display, cumulative_reward, "-yv", label='eps_greedy_dec')
        case "UCB":
            plt.plot(list_display, cumulative_reward, "-bs", label='UCB')
          
            
def plot_time(name, duree):
    plt.figure("time")
    plt.ylabel("Time (in seconds)")
    plt.xlabel("Number of pulls")
    match name:
        case "random":
            plt.plot(list_display, duree,"-r^", label='random')
        case "eps_greedy":
            plt.plot(list_display, duree, "-go", label='eps_greedy')
        case "eps_greedy_dec":
            plt.plot(list_display, duree, "-yv", label='eps_greedy_dec')
        case "UCB":
            plt.plot(list_display, duree, "-bs", label='UCB')

def bandit(K,N,name):
    s = [0 for i in range(K)]
    n = [1 for i in range(K)]
    global cumulative_reward
    global p
    for i in range(K):
        r = pull(i)
        s[i] = r
    start = time()
    for t in range(K+1,N+1):
        match name:
            case "random":
                im = randint(0, K-1)
            case "eps_greedy":
                eps = 0.1
                if random() < eps:
                    im = randint(0, K-1)
                else:
                    max = 0
                    for i in range(K):
                        temp = s[i]/n[i]
                        if temp > max:
                            max = temp
                            im = i
            case "eps_greedy_dec":
                eps = 1/log(t**2)
                if random() < eps:
                    im = randint(0, K-1)
                else:
                    max = 0
                    for i in range(K):
                        temp = s[i]/n[i]
                        if temp > max:
                            max = temp
                            im = i
            case "UCB":
                max = 0
                for i in range(K):
                    temp = s[i]/n[i] + sqrt((2*log(t) / n[i]))
                    if temp > max:
                        max = temp
                        im = i

        if(t%(N/10)) == 0:
            duree[int(t*10/N)-1] = (time()-start) + duree[int(t*10/N)-1]

        r = pull(im)
        s[im] += r
        n[im] += 1
        if(t%(N/10)) == 0:
            cumulative_reward[p] = sum(s[m] for m in range(K)) 
            p += 1      
    return sum(s[m] for m in range(K))

for i in range(number_of_try):
    bandit(K,N,"random")
plot_time("random", [t/number_of_try for t in duree]) # Avergae of each case of the list "duree"
plot_cumulative_reward("random",mean_cumulative_reward(cumulative_reward), list_display)
p = 0
for i in range(number_of_try):
    bandit(K,N,"eps_greedy")
plot_time("eps_greedy", [t/number_of_try for t in duree])
plot_cumulative_reward("eps_greedy",mean_cumulative_reward(cumulative_reward), list_display)
p = 0
for i in range(number_of_try):
    bandit(K,N,"eps_greedy_dec")
plot_time("eps_greedy_dec", [t/number_of_try for t in duree])
plot_cumulative_reward("eps_greedy_dec",mean_cumulative_reward(cumulative_reward), list_display)
p = 0
for i in range(number_of_try):
    bandit(K,N,"UCB")
plot_time("UCB", [t/number_of_try for t in duree])
plt.legend()
# plt.savefig('time', format='pdf')
plot_cumulative_reward("UCB",mean_cumulative_reward(cumulative_reward), list_display)
plt.legend()
# plt.savefig('reward', format='pdf')
plt.show()
