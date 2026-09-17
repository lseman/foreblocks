import re

import gurobipy
import gurobipy as gp
from gurobipy import GRB

instancia = {}
interesses = [
    "jobs",
    "recurso_p",
    "tamanho",
    "priority",
    "uso_p",
    "min_statup",
    "max_statup",
    "min_cpu_time",
    "max_cpu_time",
    "min_periodo_job",
    "max_periodo_job",
    "win_min",
    "win_max",
]

for interesse in interesses:
    with open("examples2/125_9.jl", "r") as exemplo:
        lines = exemplo.readlines()
        for line in lines:
            # check if string present on a current line
            if line.find(interesse) != -1:
                if interesse == "uso_p" or interesse == "recurso_p":
                    dados = re.findall(r"\d+\.\d+", line)
                    instancia[interesse] = [float(dado) for dado in dados]
                else:
                    dados = re.findall(r"\d+", line)
                    instancia[interesse] = [int(dado) for dado in dados]
    # print(f.read())
print(instancia)

# print(pato)
# from dp import pricing_dp
colunas_ = []
lb = 0
J = instancia["jobs"][0]
JOBS = J
T = instancia["tamanho"][0]
recurso_p = instancia["recurso_p"]
recurso_p = [int(i) * 10 for i in recurso_p]
print(recurso_p)

priority = instancia["priority"]  # prioridade de cada tarefa
uso_p = instancia["uso_p"]  # recurso utilizado por cada tarefa
min_statup = instancia[
    "min_statup"
]  # tempo mínimo de vezes que uma tarefa pode iniciar
max_statup = instancia[
    "max_statup"
]  # tempo máximo de vezes que uma tarefa pode iniciar
min_cpu_time = instancia[
    "min_cpu_time"
]  # tempo mínimo de unidades de tempo que uma tarefa pode consumir em sequência
max_cpu_time = instancia[
    "max_cpu_time"
]  # tempo máximo de unidades de tempo que uma tarefa pode consumir em sequência
min_periodo_job = instancia[
    "min_periodo_job"
]  # tempo mínimo que uma tarefa deve esperar para se repetir
max_periodo_job = instancia[
    "max_periodo_job"
]  # tempo máximo que uma tarefa pode esperar para se repetir
win_min = instancia["win_min"]
win_max = instancia["win_max"]


# create a model
model = gurobipy.Model()


model.Params.LogToConsole = 1
# create decision variables

x = {}
alpha = {}
soc = {}
i = {}
b = {}
phi = {}
for j in range(JOBS):
    for t in range(T):
        x[j, t] = model.addVar(
            name="x(%s,%s)" % (j, t), lb=0, ub=1, vtype=gurobipy.GRB.BINARY
        )
        phi[j, t] = model.addVar(
            vtype=gurobipy.GRB.BINARY,
            name="phi(%s,%s)" % (j, t),
        )


for t in range(T):
    alpha[t] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="alpha(%s)" % t)
    soc[t] = model.addVar(vtype=GRB.CONTINUOUS, name="soc(%s)" % t)
    i[t] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="i(%s)" % t)
    b[t] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b(%s)" % t)

soc_inicial = 0.7
limite_inferior = 0.0
ef = 0.9
v_bat = 3.6
q = 5
bat_usage = 5

# set objective
model.setObjective(
    sum(priority[j] * x[j, t] for j in range(J) for t in range(T)),
    gurobipy.GRB.MAXIMIZE,
)

# phi defines startups of jobs
for t in range(T):
    for j in range(J):
        if t == 0:
            model.addConstr(phi[j, t] >= x[j, t] - 0)
        else:
            model.addConstr(phi[j, t] >= x[j, t] - x[j, t - 1])

        model.addConstr(phi[j, t] <= x[j, t])

        if t == 0:
            model.addConstr(phi[j, t] <= 2 - x[j, t] - 0)
        else:
            model.addConstr(phi[j, t] <= 2 - x[j, t] - x[j, t - 1])

# minimum and maximum number of startups of a job
for j in range(J):
    model.addConstr(sum(phi[j, t] for t in range(T)) >= min_statup[j])
    model.addConstr(sum(phi[j, t] for t in range(T)) <= max_statup[j])

    ###############################
    # precisa ajustar

    # execution window
    model.addConstr(sum(x[j, t] for t in range(win_min[j])) == 0)
    model.addConstr(sum(x[j, t] for t in range(win_max[j], T)) == 0)

for j in range(J):
    # minimum period between jobs
    for t in range(0, T - min_periodo_job[j] + 1):
        model.addConstr(sum(phi[j, t_] for t_ in range(t, t + min_periodo_job[j])) <= 1)

    # periodo máximo entre jobs
    for t in range(0, T - max_periodo_job[j] + 1):
        model.addConstr(sum(phi[j, t_] for t_ in range(t, t + max_periodo_job[j])) >= 1)

    # min_cpu_time das jobs
    for t in range(0, T - min_cpu_time[j] + 1):
        model.addConstr(
            sum(x[j, t_] for t_ in range(t, t + min_cpu_time[j]))
            >= min_cpu_time[j] * phi[j, t]
        )

    # max_cpu_time das jobs
    for t in range(0, T - max_cpu_time[j]):
        model.addConstr(
            sum(x[j, t_] for t_ in range(t, t + max_cpu_time[j] + 1)) <= max_cpu_time[j]
        )

    # min_cpu_time no final do periodo
    for t in range(T - min_cpu_time[j] + 1, T):
        model.addConstr(sum(x[j, t_] for t_ in range(t, T)) >= (T - t) * phi[j, t])

################################
# Add power constraints
for t in range(T):
    model.addConstr(
        sum(uso_p[j] * x[j, t] for j in range(J)) <= recurso_p[t] + bat_usage * v_bat
    )  # * (1 - alpha[t]))

################################
# Bateria
################################

for t in range(T):
    model.addConstr(sum(uso_p[j] * x[j, t] for j in range(J)) + b[t] == recurso_p[t])


# Define the i_t, SoC_t constraints in Gurobi
for t in range(T):
    # P = V * I
    model.addConstr(b[t] / v_bat >= i[t])

    if t == 0:
        # SoC(1) = SoC(0) + p_carga[1]/60
        model.addConstr(soc[t] == soc_inicial + (ef / q) * (i[t] / 60))
    else:
        # SoC(t) = SoC(t-1) + (ef / Q) * I(t)
        model.addConstr(soc[t] == soc[t - 1] + (ef / q) * (i[t] / 60))

    # Set the lower and upper limits on SoC
    model.addConstr(limite_inferior <= soc[t])
    model.addConstr(soc[t] <= 1)

# valid inequalities

# first
for j in range(J):
    for t in range(T):
        model.addConstr(
            gp.quicksum(phi[j, t_] for t_ in range(t, min(T, t + min_cpu_time[j] + 1)))
            <= 1,
            name=f"VI_min_CPU_TIME_phi({j},{t})",
        )

# third
for j in range(J):
    model.addConstr(
        gp.quicksum(x[j, t] for t in range(T))
        <= max_cpu_time[j] * gp.quicksum(phi[j, t] for t in range(T)),
        name=f"VI_max_cpu_time_2({j})",
    )
# fourth
for j in range(J):
    for t in range(0, T - max_cpu_time[j], 1):
        model.addConstr(
            gp.quicksum(x[j, t_] for t_ in range(t, t + max_cpu_time[j], 1))
            <= max_cpu_time[j]
            * gp.quicksum(
                phi[j, t_]
                for t_ in range(t - max_cpu_time[j] + 1, t + max_cpu_time[j], 1)
            ),
            name=f"VI_max_cpu_time_3({j},{t})",
        )
# fifth
for j in range(J):
    for t in range(0, T - min_periodo_job[j] + 1):
        model.addConstr(
            gp.quicksum(x[j, t_] for t_ in range(t, t + min_periodo_job[j]))
            <= min_periodo_job[j],
            name=f"VI_min_period_btw_jobs_2({j},{t})",
        )

# sixth
if max_cpu_time[j] < (max_periodo_job[j] - min_cpu_time[j]):
    for t in range(0, T - max_cpu_time[j]):
        model.addConstr(phi[j, t] + x[j, t + max_cpu_time[j]] <= 1)

model.update()
# print(x)
model.optimize()
print(model.ObjVal)


import matplotlib
from matplotlib import pyplot as plt

font = {"weight": "normal", "size": 8}

matplotlib.rc("font", **font)
matplotlib.style.use("bmh")

# plt.plot([model.getVarByName('i(%s)' % t).x*v_bat for t in range(T)], label='Battery power')
# plt.plot([recurso_p[t] - model.getVarByName('b(%s)' % t).x for t in range(T)], label='Battery power')
# plt.plot([(1 - model.getVarByName('alpha(%s)' % t).x) * bat_usage * v_bat for t in range(T)], label='Battery usage')

plt.plot(recurso_p, label="Solar panel power")

task_consumption = []
for t in range(T):
    task_consumption.append(
        sum(uso_p[j] * model.getVarByName("x(%s,%s)" % (j, t)).x for j in range(J))
    )
plt.plot(task_consumption, color="gray", label="Task consumption")
# plt.plot([task_consumption[t] - recurso_p[t] for t in range(T)], label='Battery usage')
# plt.plot([model.getVarByName('soc(%s)' % t).x*100 for t in range(T)], label='SoC [%]')
plt.fill_between(
    range(T),
    recurso_p,
    task_consumption,
    color="black",
    alpha=0.05,
    label="Battery power",
)
plt.xlabel("Time [min]")
plt.ylabel("Power [W]")
plt.ylim(0, 20)
plt.legend(loc="upper left")
ax2 = plt.twinx()
ax2.plot(
    [model.getVarByName("soc(%s)" % t).x * 100 for t in range(T)], "k", label="SoC [%]"
)
ax2.set_ylim(0, 100)
ax2.legend(loc="upper right")
plt.xlim(0, len(recurso_p) - 2)

plt.savefig("ilustra.pdf", bbox_inches="tight")

plt.clf()

cmap = plt.get_cmap("viridis")

fig, axs = plt.subplots(J, 1, sharex=True, figsize=(10, 10))
for j in range(J):
    axs[j].stairs(
        [model.getVarByName("x(%s,%s)" % (j, t)).x for t in range(T)],
        label="Job %s" % j,
        linewidth=2,
        color=cmap(j / J),
    )
    ax2 = axs[j].twinx()
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, T - 1)
    axs[j].set_ylabel("Job %s" % j)
axs[-1].set_xlabel("Time [min]")
plt.savefig("ilustra2.pdf", bbox_inches="tight")
