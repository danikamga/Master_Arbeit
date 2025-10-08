import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import binom

# define parameters
d_max = 30   # maximum demand
x_max = 40    # maximum inventory level
y_max = 20    # maximum availability
pi = 5        # unit variable procurement cost
h = 1         # unit holding cost
k = 5         # fixed procurement cost
v = 20        # unit storage cost
par_pD = 0.4  # parameter p in distribution of demand
par_pY = 0.3  # parameter p in distribution of availability

'''{
    "d_max": 20,      # moderate demand
    "x_max": 30,      # moderate inventory
    "y_max": 30,      # high availability
    "pi": 5,          # moderate procurement cost
    "h": 2,           # moderate holding cost
    "k": 10,          # moderate fixed cost
    "v": 15,          # moderate shortage cost
    "par_pD": 0.5,    # moderate demand probability
    "par_pY": 0.2     # low yield/quality probability
    },#Test Instance 1: High Availability, Low Quality
    {
    "d_max": 20,
    "x_max": 30,
    "y_max": 10,      # low availability
    "pi": 5,
    "h": 2,
    "k": 10,
    "v": 15,
    "par_pD": 0.5,
    "par_pY": 0.9     # high yield/quality probability
}, #Test Instance 2: Low Availability, High Quality
{
    "d_max": 40,      # high demand
    "x_max": 50,
    "y_max": 25,      # moderate availability
    "pi": 5,
    "h": 2,
    "k": 10,
    "v": 15,
    "par_pD": 0.7,    # high demand probability
    "par_pY": 0.5     # moderate yield/quality probability
}, #Test Instance 3: High Demand, Moderate Availability and Quality
{
    "d_max": 15,
    "x_max": 20,
    "y_max": 15,
    "pi": 5,
    "h": 2,
    "k": 10,
    "v": 15,
    "par_pD": 0.5,
    "par_pY": 0.5'''
test_cases = [
    
    #Test Instance 4: Balanced Scenario'''
    # Edge Cases
    #\not_done{"d_max": 100, "x_max": 100, "y_max": 10, "pi": 5, "h": 1, "k": 5, "v": 20, "par_pD": 1.0, "par_pY": 0.5},  # High demand, low availability
    {"d_max": 10, "x_max": 20, "y_max": 100, "pi": 5, "h": 1, "k": 5, "v": 20, "par_pD": 0.5, "par_pY": 1.0},  # Low demand, high availability
    {"d_max": 10, "x_max": 20, "y_max": 10, "pi": 5, "h": 0, "k": 5, "v": 20, "par_pD": 0.5, "par_pY": 0.5},   # Zero holding cost
    {"d_max": 10, "x_max": 20, "y_max": 10, "pi": 5, "h": 1, "k": 5, "v": 0, "par_pD": 0.5, "par_pY": 0.5},    # Zero shortage cost

    #Null Lagerhaltungskosten, 
    # Null Fehlkosten, 
    # Geringe Nachfragewahrscheinlichkeit, 
    # Hohe Nachfragewahrscheinlichkeit, 
    # Geringe Ertragswahrscheinlichkeit,
    #  Hohe Ertragswahrscheinlichkeit

    # Sensitivity Analysis
    {"d_max": 10, "x_max": 20, "y_max": 10, "pi": 5, "h": 1, "k": 5, "v": 20, "par_pD": 0.1, "par_pY": 0.5},  # Low demand probability
    {"d_max": 10, "x_max": 20, "y_max": 10, "pi": 5, "h": 1, "k": 5, "v": 20, "par_pD": 0.9, "par_pY": 0.5},  # High demand probability
    {"d_max": 10, "x_max": 20, "y_max": 10, "pi": 5, "h": 1, "k": 5, "v": 20, "par_pD": 0.5, "par_pY": 0.1},  # Low yield probability
    {"d_max": 10, "x_max": 20, "y_max": 10, "pi": 5, "h": 1, "k": 5, "v": 20, "par_pD": 0.5, "par_pY": 0.9},  # High yield probability

    # Stress Test
    #Not done 
    #{"d_max": 50, "x_max": 100, "y_max": 50, "pi": 5, "h": 1, "k": 5, "v": 20, "par_pD": 0.5, "par_pY": 0.5}, # Large state/action space

    # Policy Behavior
    {"d_max": 10, "x_max": 20, "y_max": 10, "pi": 5, "h": 1, "k": 100, "v": 20, "par_pD": 0.5, "par_pY": 0.5}, # High fixed cost
    {"d_max": 10, "x_max": 20, "y_max": 10, "pi": 5, "h": 1, "k": 5, "v": 100, "par_pD": 0.5, "par_pY": 0.5},  # High shortage cost

    # Randomized Test
    {"d_max": 15, "x_max": 30, "y_max": 20, "pi": 7, "h": 2, "k": 10, "v": 15, "par_pD": 0.3, "par_pY": 0.7}, # Random parameters
]

def plot_policy(sigma, states, actions, val):
        inventory_levels = list(states)
        order_quantities = list(actions)
        x = np.arange(len(order_quantities))

        plt.figure(figsize=(10, 6))
        bars = []
        for i, s in enumerate(inventory_levels):
            probs = [sigma.get((s, q), 0) for q in order_quantities]
            bar = plt.bar(x + i*0.1, probs, width=0.1, label=f'Inv {s}')
            bars.append(bar)
            # Annotate each bar with its inventory level
            for j, p in enumerate(probs):
                if p > 0.01:  # Only annotate significant probabilities
                    plt.text(x[j] + i*0.1, p + 0.01, f"{s}", ha='center', va='bottom', fontsize=7, rotation=90)

        plt.xlabel('Bestellmenge')
        plt.ylabel('Eintritt Wahrscheinlichkeit')
        plt.xticks(x, order_quantities)
        plt.title('instanz '+val)
        plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1.25), ncol=6, fontsize=8)  # Move legend higher
        plt.subplots_adjust(top=0.8)  # Make room for legend above the plot
        plt.show()

def pD(d,d_max,par_pD):
    # probability of demand level d
    return math.comb(d_max, d) * (par_pD ** d) * ((1 - par_pD) ** (d_max - d))
    #return binom.cdf(d, d_max, par_pD)


def pY(y,y_max,par_pY):
    # probability of availability level y
    return math.comb(y_max, y) * (par_pY ** y) * ((1 - par_pY) ** (y_max - y))
    #return binom.cdf(y, y_max, par_pY)


def reward(s, a, pi, h, k, v):
    # compute reward for state s and action a
    val_x = s   # Inventory level encoded by s
    order_cost = pi * a
    holding_cost = h * max(0, val_x)
    fixed_order_cost = k if a > 0 else 0
    stockout_cost = v * max(0, -val_x)
    return -(order_cost + holding_cost + fixed_order_cost + stockout_cost)

def prob_computation(q, y, par_pY):
    if y <= q:
        return math.comb(q, y) * (par_pY ** y) * ((1-par_pY) ** (q - y))
    else:
        return 0.0

def transition_prob(s, a, s_prime, demands,availabilities,par_pY,x_max,par_pD):
    # compute transition probability from state s to s_prime given action a
    val_x = s
    val_x_prime = s_prime
    return sum(
        prob_computation(a,y,par_pY) * pD(d,demands[-1], par_pD)
        for y in availabilities
        for d in demands
        if val_x_prime == min(max(val_x, 0) + y - d, x_max)
    )

def run_model(d_max, x_max, y_max, pi, h, k, v, par_pD, par_pY, i):
    states = range(-d_max, x_max + 1)       # state indices for inventory levels
    actions = range(min(y_max, x_max) + 1)  # action indices for order quantities
    demands = range(d_max + 1)              # demand levels
    availabilities = range(y_max + 1)       # availability levels

    # define feasible actions
    A = {}
    for x in states:
        # determine all actions q satisfying condition q ≤ x_max - s + d_max
        valid_actions = [q for q in actions if q <= x_max - x + d_max]
        A[x] = valid_actions


    # create the model
    opt_mod = gp.Model("InventoryOptimization")
    opt_mod.setParam(GRB.Param.OptimalityTol, 1.0e-9)
    opt_mod.setParam(GRB.Param.FeasibilityTol, 1.0e-9)

    # variables
        # Optimisation model Variables
    g = opt_mod.addVar(lb=-GRB.INFINITY ,vtype=GRB.CONTINUOUS, name="g")  # Objective function
    b = opt_mod.addVars(states, vtype=GRB.CONTINUOUS, name="b")  # Bias function values

    opt_mod.update()


    # objective: maximize expected reward
    opt_mod.setObjective(g, GRB.MINIMIZE)

    constrs = {}

   
    # constraints
    for state in states:
        for action in actions:
            rhs = reward(state, action, pi, h, k, v) + gp.quicksum(transition_prob(state, action, next_state,availabilities=availabilities,demands=demands,par_pY=par_pY,x_max=x_max,par_pD=par_pD) * b[next_state] for next_state in states )
            #print("State: ",state,", action: ",action, ", Rhs: ",rhs)
            constr = opt_mod.addConstr(g + b[state] >= rhs, name=f"optimality_const[{state},{action}]")
            constrs[state, action] = constr 

    start_time = time.time()

    # solve the model
    opt_mod.optimize()


    sigma = {}
    performance_results = {}
    # process the results
    results = {
        "Inventory Level": [],
        "Order Quantity": [],
        "Probability": []
    }

    # Extract Results
    if opt_mod.status == GRB.OPTIMAL:

        cpu_time = time.time() - start_time
        for state in states:
            for action in actions:
                #print(val_x[state])
                sigma[state, action] = constrs[state, action].pi
        # get the optimal objective function value (minimal cost per period)
        total_cost_per_period = opt_mod.objVal
        plot_policy(sigma, states, actions, str(i))
        # expected total cost
        expected_costs = sum(reward(s, a, pi, h, k, v) * sigma[s, a] for s in states for a in actions)
        print(f"Expected cost: {-expected_costs:.4f}")
        # expected and maximal inventory level
        exp_inv = sum(x*sum(sigma[x, q] for q in A[x]) for x in states if x > 0)
        print(f"Expected inventory level: {exp_inv:.4f}")
        max_inv = np.max([x_val for x_val in states if (x_val >= 0 and sum(sigma[x_val, q] for q in A[x_val]) > 0)])
        print(f"Maximal inventory level: {max_inv:.4f}")
        # expected order quantity
        exp_ord_quant = sum(a * sigma[s, a] for s in states for a in actions)
        print(f"Expected order quantity: {exp_ord_quant:.4f}")
        # expected supply quantity
        exp_sup_quant = sum(pY(y,y_max,par_pY)*min(a, y) * sigma[s, a] for s in states for a in actions for y in availabilities)
        print(f"Expected supply quantity: {exp_sup_quant:.4f}")
        # expected and maximal shortage
        exp_short = sum(-x*sum(sigma[x, q] for q in A[x]) for x in states if x < 0)
        print(f"Expected shortage: {exp_short:.4f}")
        max_short = max(-x for x in states if (x < 0 and sum(sigma[x, q] for q in A[x] if sigma[x, q] > 0)))
        print(f"Maximal shortage: {max_short:.4f}")
        # write the results to a file
        with open("policy_availability_model_python.txt"+str(i), "w") as f:
            f.write(f'Optimal policy for availability model (Gurobi solver) \n\n')
            f.write(f"Inventory level | Order quantity | Probability\n")
            f.write(f'===============================================\n')
            for x in states:
                max_sum = 0
                va = A[x]
                for q in A[x]:
                    if max_sum < sigma[x, q]:
                        q_best = q
                        max_sum = sigma[x, q]
                    else:
                        q_best = 0
                f.write(f"         {x:>6} |         {q_best:>6} | {max_sum:.10f}\n")
            f.write(f'\nExpected total cost per period: {-total_cost_per_period:.4f}\n')
            f.write(f'Expected inventory level: {exp_inv:.4f}\n')
            f.write(f'Maximum inventory level: {max_inv}\n')
            f.write(f'Expected shortage: {exp_short:.4f}\n')
            f.write(f'Maximum shortage: {max_short:.4f}\n')
            f.write(f'Expected order quantity: {exp_ord_quant:.4f}\n')
            f.write(f'Expected supply quantity: {exp_sup_quant:.4f}\n')
            f.write(f'CPU time for optimization: {cpu_time:.4f} seconds\n')

    else:
        print("Model could not be solved to optimality.")

if __name__ == "__main__":
    i = 0
    for params in test_cases:
        print(f"Running test case: {params}")
        i += 1
        results = run_model(**params,i=i)
        print(f"Results: {results}\n")
    #results = run_model(d_max, x_max, y_max, pi, h, k, v, par_pD, par_pY,i=13)
