import numpy as np
from models_sto import ProductionStoPlanModel
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import time
from math import comb




def generate_antithetic_samples(I_A, T, num_sim, A):
    # Shape: [num_sim * 2][m_A][T]
    antithetic_samples = [[[0 for _ in range(num_sim)] for t in range(T)] for i in I_A]
    for sim in range(num_sim//2):
        for i in I_A:
            for t in range(T):
                u = np.random.uniform(0, 1)
                antithetic_samples[i][t][2*sim] = int(u * (2 * A[i][t] + 1))
                antithetic_samples[i][t][2*sim+1] = int((1 - u) * (2 * A[i][t] + 1))
    return antithetic_samples

def pY(q, availability, par_pY):
    if availability <= q:
        return comb(q, availability) * (par_pY ** availability) * ((1-par_pY) ** (q - availability))
    else:
        return 0.0

def main():
      # set up parameters
      '''T = 12   # number of periods
      n = 6    # number of products
      m = 4    # number of production factors
      q = 100  # number of samples
      m_A = 2  # number secondary factors
      I_A = range(min(m_A, m))'''
      n = 6         # number of products
      m = 6         # number of production factors
      m_A = 4       # number of secondary factors
      T = 12         # number of periods
      q = 100        # number of scenarios/samples
      I_A = range(min(m_A, m))
      # Normal demand
      mu = 10      # mean demand
      sigma = 20    # standard deviation

      # Binomial demand
      bin_mean = 20       # potential customers
      p = 0.4       # purchase probability

      #d = [np.random.binomial(n, p, T) for j in range(n)]
      #demand_normal = np.maximum(demand_normal, 0)  # truncate at 0 (no negative demand)

      #demand_binomial = np.random.binomial(n, p, N_samples)

      a, b = (0 - mu) / sigma, 20
      #demand_normal = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=T)

      np.random.seed(111)
      d = [truncnorm.rvs(a, b, loc=mu, scale=sigma, size=T) for j in range(n)]
      #d = [truncnorm.rvs(a, b, loc=mu, scale=sigma, size=2) for j in range(n)]
      '''
      plt.figure(figsize=(10,6))
      plt.hist(d, bins=40, alpha=0.6, label="Normal(100,20²)", density=True)
      plt.hist(d_bin, bins=40, alpha=0.6, label="Binomial(200,0.4)", density=True)
      plt.title("Comparison of Demand Distributions", fontsize=14)
      plt.xlabel("Demand", fontsize=12)
      plt.ylabel("Probability density", fontsize=12)
      plt.legend()
      plt.grid(alpha=0.3)
      plt.show()
      '''
      '''n=1
      m=1
      I_A = range(1)
      q=2
      T=2'''

      #d = [[np.random.randint(4, 8)*(1.1-np.sin(j+2*np.pi*t/T)) for t in range(T)] for j in range(n)]  # product demands
      p = [np.random.randint(120, 150) for _ in range(n)]   # product prices
      k = [np.random.randint(10, 20) for _ in range(n)]     # production cost
      h = [np.random.uniform(0.5, 2.5) for _ in range(n)]   # holding cost
      b = [np.random.randint(5, 10) for _ in I_A]           # procurement cost of secondary materials
      c = [np.random.randint(30, 60) for _ in I_A]          # procurement cost of corresponding primary materials
      A = [[np.random.randint(15, 65) for _ in range(T)] for _ in I_A]  # availability of secondary materials (stochastic)
      a = [[np.random.randint(0, 8) for _ in range(n)] for _ in range(m)]    # production coefficients
      I_minus_I_A = [i for i in range(m) if i not in I_A]   # set of non-secondary production factors
      R_fix = [[np.random.randint(75, 400) for _ in range(T)] for _ in I_minus_I_A]  # capacity of non-secondary production factors
      x_a = [np.random.randint(3, 5) for _ in range(n)]     # initial inventory levels of products
      R_a = [np.random.randint(10, 50) for _ in I_A]        # initial inventory levels of secondary materials

      

      np.random.seed(111)
      A_l = [[[np.random.randint(0, 2*A[i][t]+1) for _ in range(q)] for t in range(T)] for i in I_A]
      
      # antithetic variables to reduce variance
      for i in I_A:
            for t in range(T):
                  for l in range(q // 2, q):
                        A_l[i][t][l] = 2 * A[i][t] - A_l[i][t][l - q // 2]

      variance_a_l = np.var(A_l)
      #quality_fraction_l = [[[np.random.binomial(A_l[i][t][l], 0.5) for l in range(q)] for t in range(T)] for i in I_A]
      quality_fraction_l = [[[np.random.uniform(0.7, 0.95) for l in range(q)] for t in range(T)] for i in I_A]
      usable_A_l = [[[int(A_l[i][t][l] * quality_fraction_l[i][t][l]) for l in range(q)] for t in range(T)] for i in I_A]    

      # create and build model
      productionStoPlanModel = ProductionStoPlanModel(n, T, m, q, m_A, I_A, I_minus_I_A, R_fix, a, p, d, A, usable_A_l, h, k, b, c, R_a, x_a)
      productionStoPlanModel.build_model()

      # evaluate predictive master production schedule
      if productionStoPlanModel.optimize():

            #solve with stockout probability constraints
            #productionStoPlanModel.reoptimize_subject_to_service_level(service_level_target=0.95,epsilon=0.1)
            #productionStoPlanModel.save_results("results_model_sl")

            # solve without stockout probability constraints
            productionStoPlanModel.save_results("results_model")
            productionStoPlanModel.plot_per_period()
            productionStoPlanModel.plot_secondary_materials_evolution()
            productionStoPlanModel.plot_recourse_cost_per_product_per_period()
            productionStoPlanModel.plot_service_level_per_product_per_period()
            start_time = time.time()
            predictive_CM = productionStoPlanModel.model.objVal
            real_CM_avg_pred = productionStoPlanModel.simulate_schedule(num_sim=100)
            productionStoPlanModel.reoptimize_subject_to_non_anticipativity(f_star=predictive_CM, epsilon=0.1)
            productionStoPlanModel.save_results("results_model_na")
            real_CM_avg_pred_na = productionStoPlanModel.simulate_schedule(num_sim=100)
            # set epsilon to 0, if you don't want to use the model with non-anticipativity
            real_CM_avg_rolling = productionStoPlanModel.simulate_rolling_schedule(num_sim=100, epsilon=0)
            real_CM_avg_rolling_na = productionStoPlanModel.simulate_rolling_schedule(num_sim=100, epsilon=0.1)
            end_time = time.time()
            elapsed = end_time - start_time
            productionStoPlanModel.plot_per_period_after_rolling()
            #productionStoPlanModel.plot_secondary_materials_evolution()

            # show results in console 
            print(f"\n\nContribution margin predicted by sampling approximation without non-anticipativity: "
                  f"{predictive_CM}")
            print(f"Contribution margin predicted by sampling approximation with non-anticipativity: "
                  f"{predictive_CM}")
            print(f"Average realized contribution margin of sampling approximation without non-anticipativity: "
                  f"{real_CM_avg_pred}")
            print(f"Average realized contribution margin of sampling approximation with non-anticipativity: "
                  f"{real_CM_avg_pred_na}")
            print(f"Average realized contribution margin of rolling sampling approximation without non-anticipativity: "
                  f"{real_CM_avg_rolling}")
            print(f"Average realized contribution margin of rolling sampling approximation with non-anticipativity: "
                  f"{real_CM_avg_rolling_na}")
            print(f"Elapsed time: {elapsed} seconds")


      else:
            print("Could not determine predictive master production schedule")


if __name__ == "__main__":
      main()
