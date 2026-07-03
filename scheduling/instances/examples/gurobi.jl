using JuMP, Gurobi

model = Model(with_optimizer(Gurobi.Optimizer, OutputFlag = 1, TimeLimit = 300))


include("291_18_NADIR.jl")



println("T = ", T, "; J = ",jobs) 



soc_inicial = 0.75 # [%] de carga (0-1)
 
q = 10# [A hora] capacidade de carga da bateria
v_bat = 3.6 # [V] tensão nominal da bateria@

bat_usage = 1.5 # [A]
limite_inferior = 0.5
# ef charge discharge efficiency
ef = 0.9 # 0.95 no sol, 50 °C # 0.8 na sombra, 0°C

# d = 0.01 # delta soc_final
k = 1 
 d=0.05

@variable(model, x[1:subs, 1:jobs, 1:T], binary = true)
@variable(model, phi[1:subs, 1:jobs, 1:T], binary = true)
@variable(model, soc[1:T])
@variable(model, 0 <= alpha[1:T] <= 1)
@variable(model, b[1:T])
@variable(model, i[1:T])



for t in 1:T
    @constraint(model, b[t] / v_bat >= i[t])  # P = V * I 
    @constraint(model, b[t] == recurso_p[t] - sum(uso_p[job] * x[1,job,t] for job in 1:jobs)) # Pin(t) - Putilizado(t) = Pcarga da bateria(t)

    if t == 1
        @constraint(model, soc[t] ==  soc_inicial + (ef / q) * (i[t] / 60) ) # SoC(1) = SoC(0) + p_carga[1]/60
    else
        @constraint(model, soc[t] ==  soc[t - 1] + (ef / q) * (i[t] / 60)) # SoC(t) = SoC(t-1) + (ef / Q) * I(t)
    end

    @constraint(model, limite_inferior <= (soc[t] ) <= 1)

end

@constraint(model, (soc_inicial - soc_inicial * d) <= soc[T] <= (soc_inicial + soc_inicial * d) )


# define que o recurso utilizado deve ser menor que o recurso disponível
for t in 1:T
    @constraint(model, sum(uso_p[job] * x[1,job,t] for job in 1:jobs) <= recurso_p[t]  + bat_usage * v_bat * (1 - alpha[t]))
end



@objective(model, Max, sum(priority[job] * x[1,job,t] for job in 1:jobs for t in 1:T))
# define que o recurso utilizado deve ser menor que o recurso disponível




# phi define startups de jobs
for t in 1:T
    for job in 1:jobs
        if t == 1
            @constraint(model, phi[1, job, t] >= x[1,job,t] - 0)
        else
            @constraint(model, phi[1, job, t] >= x[1,job,t] - x[1,job,t - 1])
        end
    end
end
for t in 1:T
    for job in 1:jobs
        @constraint(model, phi[1, job, t] <= x[1,job,t])
    end
end
for t in 1:T
    for job in 1:jobs
        if t == 1
            @constraint(model, phi[1, job, t] <= 2 - x[1,job,t] - 0)
        else
            @constraint(model, phi[1, job, t] <= 2 - x[1,job,t] - x[1,job,t - 1])
        end
    end
end

# minimo e maximo numero de startups de uma job
for job in 1:jobs
    @constraint(model, sum(phi[1, job, t] for t in 1:T) >= min_statup[job])
    @constraint(model, sum(phi[1, job, t] for t in 1:T) <= max_statup[job])
end

# janela de execução
for job in 1:jobs
    @constraint(model, sum(x[1, job, t] for t in 1:win_min[job]) == 0)
    @constraint(model, sum(x[1, job, t] for t in win_max[job] + 1:T) == 0)
end


# periodo mínimo entre jobs
for job in 1:jobs
    for t in 1:T - min_periodo_job[job] + 1
        @constraint(model, sum(phi[1, job, t_] for t_ in t:t + min_periodo_job[job] - 1) <= 1)
    end
end

# periodo máximo entre jobs
for job in 1:jobs
    for t in 1:T - max_periodo_job[job] + 1
        @constraint(model, sum(phi[1, job, t_] for t_ in t:t + max_periodo_job[job] - 1) >= 1)
    end
end

# min_cpu_time das jobs
for job in 1:jobs
    for t in 1:T - min_cpu_time[job] + 1
        @constraint(model, sum(x[1, job, t_] for t_ in t:t + min_cpu_time[job] - 1) >= min_cpu_time[job] * phi[1, job, t])
    end
end

for job in 1:jobs
# max_cpu_time das jobs
    for t in 1:T - max_cpu_time[job]
        @constraint(model, sum(x[1, job, t_] for t_ in t:t + max_cpu_time[job]) <= max_cpu_time[job])
    end

# min_cpu_time no final do periodo
    for t in T - min_cpu_time[job] + 2:T
        @constraint(model, sum(x[1, job, t_] for t_ in t:T) >= (T - t + 1) * phi[1, job, t])
    end    
end    



@time begin
    JuMP.optimize!.(model);
end

objetivo = JuMP.objective_value.(model)
println("The Objective Value is: ", objetivo)
println("Solving time (min): ", solve_time(model)/60)
println("T = ", T, "; J = ",jobs) 
