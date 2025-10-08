import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import os
import matplotlib.pyplot as plt


class ProductionStoPlanModel:
    def __init__(self, n, T, m, q, m_A, I_A, I_minus_I_A, R_fix, a, p, d, A, A_l, h, k, b, c, R_a, x_a,beta_service_level_target=0.40):
        # model
        self.model = gp.Model("MPS_CE_Sampling")
        self.model.params.OptimalityTol = 1e-9
        self.model.params.FeasibilityTol = 1e-9
        self.model.params.Method = 0  # set solver to primal simplex
        self.model.params.LPWarmStart = 2  # enforce warm starts
        # parameters
        self.n = n  #
        self.T = T   #
        self.m_A = m_A   #
        self.m = m   #
        self.q = q   #
        # sets
        self.I_A = I_A   #
        self.I_minus_I_A = I_minus_I_A   #
        self.a = a   #
        self.R_fix = R_fix   #
        self.d = d   # 
        self.A = A   #
        self.A_l = A_l   # 
        self.h = h   #
        self.k = k   #
        self.b = b   #
        self.c = c   #
        self.R_a = R_a   #
        self.x_a = x_a   #
        self.p = p   #
        self.beta_service_level_target = beta_service_level_target
       # M = max(max(d))

        # decision variables
        self.x = None   #Inventory level
        self.y = None   #Produced quantity
        self.z = None   #Sold quantity
        self.v = None   #Procurement of secondary material
        self.w = None   #Procurement of primary material
        self.R = None   #Inventory level of secondary material
        #recourse
        self.period_recourse_cost = []
        self.total_period_costs = []
        self.R_mean = [[0.0] * self.T] * self.m_A
        self.v_mean = [[0.0] * self.T for _ in self.I_A]
        self.w_mean = [[0.0] * self.T for _ in self.I_A]
        #non anticipativity var
        self.x_na = [[0.0 for _ in range(self.T+1)] for _ in range(self.n)]
        self.y_na = [[0.0 for _ in range(self.T)] for _ in range(self.n)]
        self.z_na = [[0.0 for _ in range(self.T)] for _ in range(self.n)]
        self.R_na = [[0.0 for _ in range(self.T+1)] for _ in range(self.m_A)]
        self.v_na = [[0.0 for _ in range(self.T)] for _ in self.I_A]
        self.w_na = [[0.0 for _ in range(self.T)] for _ in self.I_A]
        self.beta_service_level = []
        self.service_levels = []


    # build model
    def build_model(self):
        # define variables
        # xjt: inventory level of product j at the end of period t
        # yjt: quantity of product j manufactured in period t
        # zjt: sales of product j in period t
        # Ritl: inventory level of secondary material i in period t in sample l
        # vitl: procurement of secondary material i in period t in sample l
        # witl: procurement of primary material replacing secondary material i in period t in sample l
        self.x = self.model.addVars(self.n, self.T + 1, name="x", vtype=GRB.CONTINUOUS)
        self.y = self.model.addVars(self.n, self.T, name="y", vtype=GRB.CONTINUOUS)
        self.z = self.model.addVars(self.n, self.T, name="z", vtype=GRB.CONTINUOUS)
        self.v = self.model.addVars(self.m_A, self.T, self.q, name="v", vtype=GRB.CONTINUOUS)
        self.w = self.model.addVars(self.m_A, self.T, self.q, name="w", vtype=GRB.CONTINUOUS)
        self.R = self.model.addVars(self.m_A, self.T + 1, self.q, name="R", vtype=GRB.CONTINUOUS)

        # objective function: maximize profit
        self.model.setObjective(
            gp.quicksum(
                gp.quicksum(self.p[j]*self.z[j, t] - self.k[j]*self.y[j, t] - self.h[j]*self.x[j, t+1]
                            for j in range(self.n))
                - (1/self.q)*gp.quicksum(gp.quicksum(self.b[i]*self.v[i, t, l] + self.c[i]*self.w[i, t, l]
                                                     for i in self.I_A) for l in range(self.q))
                for t in range(self.T)), GRB.MAXIMIZE)
    
        # add constraints
        self._add_constraints()

    def _add_constraints(self):
        # resource constraint for non-secondary production factors
        for t in range(self.T):
            for i in self.I_minus_I_A:
                self.model.addConstr(
                    quicksum(self.a[i][j] * self.y[j, t] for j in range(self.n)) <= self.R_fix[i-self.m_A][t],
                    name=f"ResourceConstraint_{i}_{t}")

        # inventory initialization
        for j in range(self.n):
            self.model.addConstr(self.x[j, 0] == self.x_a[j], name=f"InventoryInitProduct_{j}")

        # Service level constraints
        for j in range(self.n):
            self.model.addConstr(quicksum(self.z[j, t] for t in range(self.T)) >= quicksum(self.d[j][t] for t in range(self.T)) * self.beta_service_level_target, name=f"serviceLevelProduct_{j}")

        # initial inventory secondary material
        for l in range(self.q):
            for i in self.I_A:
                self.model.addConstr(self.R[i, 0, l] == self.R_a[i], name=f"InventoryInitSecondary_{i}_{l}")

        # sales constraint, inventory balance, non-negativity constraints
        for t in range(self.T):
            for j in range(self.n):
                self.model.addConstr(self.x[j, t + 1] == self.x[j, t] + self.y[j, t] - self.z[j, t],
                                     name=f"InventoryBalanceProduct_{j}_{t}")
                self.model.addConstr(self.z[j, t] <= self.d[j][t], name=f"SalesConstraint_{j}_{t}")

        # inventory balance constraint secondary material, non-negativity of secondary material and
        # availability constraint secondary material
        for l in range(self.q):
            for t in range(self.T):
                for i in self.I_A:
                    self.model.addConstr(
                        self.R[i, t + 1, l] == self.R[i, t, l] + self.v[i, t, l] + self.w[i, t, l] -
                        quicksum(self.a[i][j] * self.y[j, t] for j in range(self.n)),
                        name=f"InventoryBalanceSecondary_{i}_{t}_{l}"
                    )
                    self.model.addConstr(self.v[i, t, l] <= self.A_l[i][t][l],
                                         name=f"AvailabilityConstraint_{i}_{t}_{l}")
    
    def restore_model(self):
        # reset variable bounds for each simulation
        for t in range(self.T):
            for j in range(self.n):
                self.x[j, t+1].LB = 0.0
                self.x[j, t+1].UB = np.inf
                self.y[j, t].LB = 0.0
                self.y[j, t].UB = np.inf
                self.z[j, t].LB = 0.0
                self.z[j, t].UB = np.inf

            for l in range(self.q):
                for i in self.I_A:
                    self.R[i, t+1, l].LB = 0.0
                    self.R[i, t+1, l].UB = np.inf
                    self.v[i, t, l].LB = 0.0
                    self.v[i, t, l].UB = np.inf
                    self.w[i, t, l].LB = 0.0
                    self.w[i, t, l].UB = np.inf

        # add deleted constraints
        for t in range(self.T):
            for i in self.I_minus_I_A:
                self.model.addConstr(
                    gp.quicksum(self.a[i][j]*self.y[j, t] for j in range(self.n)) <= self.R_fix[i-self.m_A][t],
                    name=f"ResourceConstraint_{i}_{t}")

            for j in range(self.n):
                self.model.addConstr(self.x[j, t+1] == self.x[j, t] + self.y[j, t] - self.z[j, t],
                                     name=f"InventoryBalanceProduct_{j}_{t}")
                self.model.addConstr(self.z[j, t] <= self.d[j][t], name=f"SalesConstraint_{j}_{t}")

            for l in range(self.q):
                for i in self.I_A:
                    self.model.addConstr(
                        self.R[i, t + 1, l] == self.R[i, t, l] + self.v[i, t, l] + self.w[i, t, l] -
                        gp.quicksum(self.a[i][j] * self.y[j, t] for j in range(self.n)),
                        name=f"InventoryBalanceSecondary_{i}_{t}_{l}")
                    self.model.addConstr(self.v[i, t, l] <= self.A_l[i][t][l],
                                         name=f"AvailabilityConstraint_{i}_{t}_{l}")

        self.model.update()

    def reoptimize_subject_to_non_anticipativity(self, f_star, epsilon):
        self.model.addConstr(gp.quicksum(gp.quicksum(
            self.p[j]*self.z[j, t] - self.k[j]*self.y[j, t] - self.h[j]*self.x[j, t+1] for j in range(self.n))
            - (1/self.q)*gp.quicksum(gp.quicksum(
                self.b[i]*self.v[i, t, l] + self.c[i]*self.w[i, t, l] for i in self.I_A) for l in range(self.q))
                                         for t in range(self.T)) == f_star, name="OptimalityConstraint")
        self.model.setObjective(gp.quicksum((1+epsilon)**t * gp.quicksum(gp.quicksum(
            self.v[i, t, l] for i in self.I_A) for l in range(self.q))
                                            for t in range(self.T)), GRB.MINIMIZE)
        self.model.optimize()
        # reset model to original model
        self.model.setObjective(gp.quicksum(gp.quicksum(
            self.p[j]*self.z[j, t] - self.k[j]*self.y[j, t] - self.h[j]*self.x[j, t+1] for j in range(self.n))
                    - (1/self.q)*gp.quicksum(gp.quicksum(self.b[i]*self.v[i, t, l] + self.c[i]*self.w[i, t, l]
                                                         for i in self.I_A) for l in range(self.q))
                                            for t in range(self.T)), GRB.MAXIMIZE)
        self.model.remove(self.model.getConstrByName(name="OptimalityConstraint"))

    def reoptimize_subject_to_service_level(self, service_level_target, epsilon):

        M = max(max(self.d))
        # add service level constraints
        for j in range(self.n):
            for t in range(self.T):
                beta_jt = self.model.addVar(name=f"beta_service_level_{j}_{t}", vtype=GRB.BINARY)
                self.beta_service_level.append(beta_jt)
                self.model.addConstr(self.z[j, t] + M * beta_jt >= self.d[j][t],
                                     name=f"ServiceLevelConstraint_{j}_{t}")
        self.model.setObjective(gp.quicksum(gp.quicksum(
            self.p[j]*self.z[j, t] - self.k[j]*self.y[j, t] - self.h[j]*self.x[j, t+1] for j in range(self.n))
                    - (1/self.q)*gp.quicksum(gp.quicksum(self.b[i]*self.v[i, t, l] + self.c[i]*self.w[i, t, l]
                                                         for i in self.I_A) for l in range(self.q))
                                            for t in range(self.T)), GRB.MAXIMIZE)
        #self.model.addConstr(gp.quicksum(self.beta_service_level) <= self.n * self.T * (1 - service_level_target),
                             #name="TotalServiceLevelConstraint")
        self.model.optimize()
        # reset model to original model
        self.model.setObjective(gp.quicksum(gp.quicksum(
            self.p[j]*self.z[j, t] - self.k[j]*self.y[j, t] - self.h[j]*self.x[j, t+1] for j in range(self.n))
                    - (1/self.q)*gp.quicksum(gp.quicksum(self.b[i]*self.v[i, t, l] + self.c[i]*self.w[i, t, l]
                                                         for i in self.I_A) for l in range(self.q))
                                            for t in range(self.T)), GRB.MAXIMIZE)
        for j in range(self.n):
            for t in range(self.T):
                self.model.remove(self.model.getConstrByName(name=f"ServiceLevelConstraint_{j}_{t}"))
        self.model.remove(self.model.getConstrByName(name="TotalServiceLevelConstraint"))
        self.beta_service_level = []
        
    def optimize(self):
        self.model.optimize()
        return self.model.status == GRB.OPTIMAL

    def simulate_rolling_schedule(self, num_sim, epsilon):
        real_CMs_rolling = []
        for ctr in range(num_sim):
            np.random.seed(ctr+1)
                
            CM_without_secondary_materials_cost = 0.0
            secondary_materials_cost = 0.0

            # iterations of rolling horizon approach
            for tau in range(self.T):
                # solve model with decisions fixed up to tau-1
                if self.optimize():
                    if epsilon > 0.0:
                        f_star = self.model.objVal
                        self.reoptimize_subject_to_non_anticipativity(f_star, epsilon)
                    # fix variables at time tau
                    for j in range(self.n):
                        # retrieve x, y, and z values
                        x_value = self.x[j, tau].x
                        y_value = self.y[j, tau].x
                        z_value = self.z[j, tau].x
                        new_x_value = max(0.0, x_value + y_value - z_value)  # max due to inaccuracies

                        # fix x, y, and z variables
                        self.x[j, tau+1].LB = new_x_value
                        self.x[j, tau+1].UB = new_x_value
                        self.y[j, tau].LB = y_value
                        self.y[j, tau].UB = y_value
                        self.z[j, tau].LB = z_value
                        self.z[j, tau].UB = z_value

                        self.x_na[j][tau+1] = new_x_value
                        self.y_na[j][tau] = y_value
                        self.z_na[j][tau] = z_value

                        CM_without_secondary_materials_cost += self.p[j]*z_value - self.k[j]*y_value \
                            - self.h[j]*new_x_value
                       
                    for i in self.I_A:
                        # retrieve R value
                        R_value = self.R[i, tau, 0].x

                        # sample availability
                        realized_A = np.random.randint(0, 2*self.A[i][tau]+1)

                        # compute realized purchases of secondary and primary materials
                        sum_req = sum([self.a[i][j] * self.y[j, t].x for t in range(tau, self.T)
                                       for j in range(self.n)])
                        if R_value + realized_A <= sum_req:
                            realized_v = realized_A
                        else:
                            realized_v = max(0.0, sum_req - R_value)

                        realized_w = max(0.0, sum([self.a[i][j]*self.y[j, tau].x for j in range(self.n)])
                                         - R_value - realized_v)

                        secondary_materials_cost += (self.b[i] * realized_v + self.c[i] * realized_w)

                        self.R_na[i][tau+1] = R_value + realized_v + realized_w - sum([self.a[i][j]*self.y[j, tau].x
                                                                                     for j in range(self.n)])
                        self.v_na[i][tau] = realized_v
                        self.w_na[i][tau] = realized_w

                        # fix purchase variables and inventory
                        for l in range(self.q):
                            self.v[i, tau, l].LB = realized_v
                            self.v[i, tau, l].UB = realized_v
                            self.w[i, tau, l].LB = realized_w
                            self.w[i, tau, l].UB = realized_w
                            new_R_value = R_value + realized_v + realized_w - sum([self.a[i][j]*self.y[j, tau].x
                                                                                   for j in range(self.n)])
                            self.R[i, tau+1, l].LB = new_R_value
                            self.R[i, tau+1, l].UB = new_R_value

                    # remove constraints for time tau
                    for l in range(self.q):
                        for i in self.I_A:
                            self.model.remove(self.model.getConstrByName(f"InventoryBalanceSecondary_{i}_{tau}_{l}"))
                            self.model.remove(self.model.getConstrByName(f"AvailabilityConstraint_{i}_{tau}_{l}"))
                        
                    for i in self.I_minus_I_A:
                        self.model.remove(self.model.getConstrByName(f"ResourceConstraint_{i}_{tau}"))

                    for j in range(self.n):
                        self.model.remove(self.model.getConstrByName(f"InventoryBalanceProduct_{j}_{tau}"))
                        self.model.remove(self.model.getConstrByName(f"SalesConstraint_{j}_{tau}"))    
                    
                    self.model.update()

            total_CM = CM_without_secondary_materials_cost - secondary_materials_cost
            real_CMs_rolling.append(total_CM)
            self.restore_model()
    
        real_CM_avg_rolling = np.mean(real_CMs_rolling)
        return real_CM_avg_rolling

    def simulate_schedule(self, num_sim):
        real_CMs = []
        for ctr in range(num_sim):
            # initialize
            np.random.seed(ctr+1)
            CM_without_secondary_materials_cost = sum([self.p[j]*self.z[j, t].x-self.k[j]*self.y[j, t].x
                                                       - self.h[j]*self.x[j, t+1].x for j in range(self.n)
                                                       for t in range(self.T)])
            secondary_materials_cost = 0.0
            R_values = [self.R_a[i] for i in self.I_A]

            # iterate periods
            for tau in range(self.T):
                for i in self.I_A:
                    # sample availability
                    realized_A = np.random.randint(0, 2*self.A[i][tau]+1)
                    R_value = R_values[i]

                    # compute realized purchases of secondary and primary materials
                    sum_req = sum([self.a[i][j] * self.y[j, t].x for t in range(tau, self.T)
                                   for j in range(self.n)])
                    if R_value + realized_A <= sum_req:
                        realized_v = realized_A
                    else:
                        realized_v = max(0.0, sum_req - R_value)
                    realized_w = max(0.0, sum([self.a[i][j] * self.y[j, tau].x for j in range(self.n)])
                                     - R_value - realized_v)

                    # update inventory for tau + 1
                    R_values[i] = R_value + realized_v + realized_w - sum([self.a[i][j] * self.y[j, tau].x
                                                                           for j in range(self.n)])
                    secondary_materials_cost += self.b[i] * realized_v + self.c[i] * realized_w

            total_CM = CM_without_secondary_materials_cost - secondary_materials_cost
            real_CMs.append(total_CM)

        real_CM = np.mean(real_CMs)
        return real_CM

    def save_results(self, filename):

        # check whether folder results exists; if not, create folder
        if not os.path.isdir("./OM_for_Reverse_Supply_Chains-main/master_production_scheduling/Python/stochastic/code/results"):
            os.makedirs("./OM_for_Reverse_Supply_Chains-main/master_production_scheduling/Python/stochastic/code/results")
        with open(f"./OM_for_Reverse_Supply_Chains-main/master_production_scheduling/Python/stochastic/code/results/{filename}.txt", "w") as f:
            # Summary of key metrics
            f.write("Optimal circular master production schedule\n\n")
            f.write("------------------------------------------------------------\n")
            f.write(" Product | Period | Inventory | Production | Sales | Demand \n")
            f.write("------------------------------------------------------------\n")
            for i in range(self.n):
                for t in range(self.T):
                    # Write rows for each product i and period t
                    f.write(f"{i+1:8} | {t+1:6} | {self.x[i,t].x:9.2f} | {self.y[i,t].x:10.2f} | {self.z[i,t].x:5.2f} "
                            f"| {self.d[i][t]:6.2f} \n")
            f.write("\n")
            f.write("---------------------------------------------------------------------------------\n")
            f.write(" Secondary m. | Period |  Inventory | Procurement sec. m. | Procurement prim. m. \n")
            f.write("---------------------------------------------------------------------------------\n")
            
            for i in self.I_A:
                for t in range(self.T):
                    self.v_mean[i][t] = (1/self.q) * sum(self.v[i, t, l].x for l in range(self.q))
            
            for i in self.I_A:
                for t in range(self.T):
                    self.w_mean[i][t] = (1/self.q) * sum(self.w[i, t, l].x for l in range(self.q))
            
            
            for i in self.I_A:
                for t in range(self.T):
                    self.R_mean[i][t] = (1/self.q)*sum(self.R[i, t, l].x for l in range(self.q))

            
            #Compute the recourse cost per period
            for t in range(self.T):    
                recourse = sum(self.b[i] * self.v_mean[i][t] + self.c[i] * self.w_mean[i][t] for i in self.I_A)
                self.period_recourse_cost.append(recourse)

            for i in self.I_A:
                for t in range(self.T):
                    # Write rows for each secondary material i and period t
                    f.write(f"{i+1:13} | {t+1:6} | {self.R_mean[i][t]:10.2f} | {self.v_mean[i][t]:19.2f} "
                            f"| {self.w_mean[i][t]:20.2f} \n")

            f.write("\n")

            #Compute the bêta service level for each product
            
            for j in range(self.n):
                #beta_service_level_j = sum(min(self.x[j, t].x + self.y[j, t].x, self.d[j][t]) for t in range(self.T))/sum(self.d[j][t] for t in range(self.T))
                beta_service_level_j = 0
                sum_zt = 0
                sum_dt = 0
                for t in range(self.T):
                    sum_zt += self.z[j, t].x 
                    sum_dt += self.d[j][t] 
                beta_service_level_j = sum_zt / sum_dt
                self.beta_service_level.append(beta_service_level_j)

            # Compute alpha service level for each product
            
            for j in range(self.n):
                # Check if inventory + production >= demand in each period
                fulfilled_periods = sum((self.x[j, t].x + self.y[j, t].x) >= self.d[j][t] for t in range(self.T))
                service_level_j = fulfilled_periods / self.T  # fraction of periods with fulfilled demand
                self.service_levels.append(service_level_j)

            for j, sl in enumerate(self.service_levels):
                f.write(f"Service level for product {j + 1}: {sl:.4f}\n")

            TCM = sum(self.p[j] * self.z[j, t].X - self.k[j] * self.y[j, t].X - self.h[j] * self.x[j, t + 1].X
                      for j in range(self.n) for t in range(self.T)) \
                - sum(self.b[i] * self.v_mean[i][t] + self.c[i] * self.w_mean[i][t]
                      for i in self.I_A for t in range(self.T))

            f.write(f"Total contribution margin: {TCM:.4f}\n")


    def plot_per_period_after_rolling(self):
        for j in range(self.n):
            periods = list(range(1, self.T + 1))
            inventory = [self.x_na[j][t] for t in range(1,self.T+1)]
            production = self.y_na[j]
            sales = self.z_na[j]
            demand = self.d[j]

            plt.figure(figsize=(10, 6))
            plt.plot(periods, inventory, label='Inventory', marker='o')
            plt.plot(periods, production, label='Production', marker='s')
            plt.plot(periods, sales, label='Sales', marker='^')
            plt.plot(periods, demand, label='Demand', marker='x')
            plt.title(f'Product {j+1}: Demand, Inventory, Production, Sales per Period')
            plt.xlabel('Period')
            plt.ylabel('Quantity')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_per_period(self):
        for j in range(self.n):
            periods = list(range(1, self.T + 1))
            inventory = [self.x[j, t].x for t in range(1, self.T + 1)]
            production = [self.y[j, t].x for t in range(self.T)]
            sales = [self.z[j, t].x for t in range(self.T)]
            demand = [self.d[j][t] for t in range(self.T)]

            plt.figure(figsize=(10, 6))
            plt.plot(periods, inventory, label='Inventory', marker='o')
            plt.plot(periods, production, label='Production', marker='s')
            plt.plot(periods, sales, label='Sales', marker='^')
            plt.plot(periods, demand, label='Demand', marker='x')
            plt.title(f'Product {j+1}: Demand, Inventory, Production, Sales per Period')
            plt.xlabel('Period')
            plt.ylabel('Quantity')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_recourse_cost_per_product_per_period(self):
        """
        Plots the recourse cost for each secondary material (product) per period.
        """
        periods = list(range(1, self.T + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(periods, self.period_recourse_cost, marker='o', label=f'period recourse costs')
        plt.title('Recourse Cost per Period')
        plt.xlabel('Period')
        plt.ylabel('Recourse Cost')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_secondary_materials_evolution(self):
        """
        Plots the evolution of v_mean (secondary procurement), w_mean (primary procurement),
        and R_mean (inventory) per secondary material over periods.
        """
        periods = list(range(1, self.T + 1))

        for i in self.I_A:
            plt.figure(figsize=(10, 6))
            plt.plot(periods, self.v_mean[i], marker='o', label='Secondary Procurement (v_mean)')
            plt.plot(periods, self.w_mean[i], marker='s', label='Primary Procurement (w_mean)')
            plt.plot(periods, self.R_mean[i], marker='^', label='Inventory (R_mean)')
            plt.title(f'Evolution of Secondary Material {i+1} Over Periods')
            plt.xlabel('Period')
            plt.ylabel('Quantity')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_service_level_per_product_per_period(self):
        """
        Plots the service level (sales/demand) for each product per period in one graph.
        """
        products = list(range(1, self.n + 1))
        plt.figure(figsize=(10, 6))
        
        plt.plot(products, self.service_levels, marker='o', label=f'alpha Service Level')
        plt.plot(products, self.beta_service_level, marker='x', linestyle='--', label=f'Beta Service Level Product ')
        plt.title('Service Level per Product')
        plt.xlabel('Product')
        plt.ylabel('Service Level (Sales / Demand)')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()