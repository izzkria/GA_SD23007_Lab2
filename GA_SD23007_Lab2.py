import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---- GA Parameters ----
POP_SIZE = 300
CHROM_LEN = 80
TARGET_ONES = 50
MAX_GEN = 50

# ---- GA Functions ----
def create_population():
    return np.random.randint(2, size=(POP_SIZE, CHROM_LEN))

def fitness(pop):
    return np.sum(pop, axis=1)

def select(pop, fit):
    idx = np.argsort(fit)[-POP_SIZE//2:]
    return pop[idx]

def crossover(pop):
    new_pop = []
    for i in range(0, len(pop), 2):
        p1, p2 = pop[i], pop[i+1]
        point = random.randint(1, CHROM_LEN-1)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        new_pop.extend([c1, c2])
    return np.array(new_pop)

def mutate(pop, rate=0.01):
    for p in pop:
        for i in range(CHROM_LEN):
            if random.random() < rate:
                p[i] = 1 - p[i]
    return pop

# ---- Streamlit UI ----
st.title("Genetic Algorithm Demo")

pop = create_population()
avg_fit_list = []

for gen in range(MAX_GEN):
    fit = fitness(pop)
    avg_fit = np.mean(fit)
    avg_fit_list.append(avg_fit)
    
    pop = select(pop, fit)
    pop = crossover(pop)
    pop = mutate(pop)

st.subheader("GA Average Fitness Over Generations")
fig, ax = plt.subplots()
ax.plot(range(MAX_GEN), avg_fit_list, marker='o')
ax.set_xlabel("Generation")
ax.set_ylabel("Average Fitness")
st.pyplot(fig)

