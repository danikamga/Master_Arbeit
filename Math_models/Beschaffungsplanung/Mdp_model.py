import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
import matplotlib.pyplot as plt
import time

'''d_max = 30   # maximum demand
x_max = 40    # maximum inventory level
y_max = 20    # maximum availability
pi = 5        # unit variable procurement cost
h = 1         # unit holding cost
k = 5         # fixed procurement cost
v = 10        # unit storage cost
par_pD = 0.4  # parameter p in distribution of demand
par_pY = 0.3  # parameter p in distribution of availability'''

'''# Test Instance 1: High Availability, Low Quality
d_max = 20      # moderate demand
x_max = 30      # moderate inventory
y_max = 30      # high availability
pi = 5          # moderate procurement cost
h = 2           # moderate holding cost
k = 10          # moderate fixed cost
v = 15          # moderate shortage cost
par_pD = 0.5    # moderate demand probability
par_pY = 0.2    # low yield/quality probability

# Test Instance 2: Low Availability, High Quality
d_max = 20
x_max = 30
y_max = 10      # low availability
pi = 5
h = 2
k = 10
v = 15
par_pD = 0.5
par_pY = 0.9    # high yield/quality probability

# Test Instance 3: High Demand, Moderate Availability and Quality
d_max = 40      # high demand
x_max = 50
y_max = 25      # moderate availability
pi = 5
h = 2
k = 10
v = 15
par_pD = 0.7    # high demand probability
par_pY = 0.5    # moderate yield/quality probability'''

# Test Instance 4: Balanced Scenario
d_max = 15
x_max = 20
y_max = 15
pi = 5
h = 2
k = 10
v = 15
par_pD = 0.5
par_pY = 0.5
 


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

def plot_policy(sigma, states, actions):
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
    plt.title('Optimale Bestellpolitik')
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1.25), ncol=6, fontsize=8)  # Move legend higher
    plt.subplots_adjust(top=0.8)  # Make room for legend above the plot
    plt.show()

def pD(d):
    # probability of demand level d
    return math.comb(d_max, d) * (par_pD ** d) * ((1 - par_pD) ** (d_max - d))


def pY(y):
    # probability of availability level y
    return math.comb(y_max, y) * (par_pY ** y) * ((1 - par_pY) ** (y_max - y))


def reward(s, a):
    # compute reward for state s and action a
    val_x = s   # Inventory level encoded by s
    order_cost = pi * sum(pY(y) * min(a, y) for y in availabilities)
    holding_cost = h * max(0, val_x)
    fixed_order_cost = k if a > 0 else 0
    stockout_cost = v * max(0, -val_x)
    return -(order_cost + holding_cost + fixed_order_cost + stockout_cost)


def transition_prob(s, a, s_prime):
    # compute transition probability from state s to s_prime given action a
    val_x = s
    val_x_prime = s_prime
    return sum(
        pY(y) * pD(d)
        for y in availabilities
        for d in demands
        if val_x_prime == min(max(val_x, 0) + min(a, y) - d, x_max)
    )

# objective: maximize expected reward
opt_mod.setObjective(g, GRB.MINIMIZE)

constrs = {}

p=[]
for s in states:
    for a in A[s]:
        for s_prime in states:
            p.append(transition_prob(s, a, s_prime))

reward_values = []
for s in states:
    for a in A[s]:
        reward_values.append(reward(s, a))
# constraints
for state in states:
    for action in actions:
        rhs = reward(state, action) + gp.quicksum(transition_prob(state, action, next_state) * b[next_state] for next_state in states )
        #print("State: ",state,", action: ",action, ", Rhs: ",rhs)
        constr = opt_mod.addConstr(g + b[state] >= rhs, name=f"optimality_const[{state},{action}]")
        constrs[state, action] = constr 

'''for x in states:
    opt_mod.addConstr(g + b[x] <= gp.quicksum(
        reward(x, q) * gp.quicksum(
            transition_prob(x, q, next_state) * b[next_state] for next_state in states
        ) for q in A[x]
    ), name=f"constraint_{x}")'''
    

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
    plot_policy(sigma, states, actions)
    # expected total cost
    expected_costs = sum(reward(s, a) * sigma[s, a] for s in states for a in actions)
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
    exp_sup_quant = sum(pY(y)*min(a, y) * sigma[s, a] for s in states for a in actions for y in availabilities)
    print(f"Expected supply quantity: {exp_sup_quant:.4f}")
    # expected and maximal shortage
    exp_short = sum(-x*sum(sigma[x, q] for q in A[x]) for x in states if x < 0)
    print(f"Expected shortage: {exp_short:.4f}")
    max_short = max(-x for x in states if (x < 0 and sum(sigma[x, q] for q in A[x] if sigma[x, q] > 0)))
    print(f"Maximal shortage: {max_short:.4f}")
    # write the results to a file
    with open("policy_availability_model_python.txt", "w") as f:
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
