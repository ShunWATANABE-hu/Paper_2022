import csv
import datetime
import math
import numpy as np
import time

parameter = [[i,j] for i in np.arange(0.1,1.0,0.1) for j in np.arange(0.1,1.0,0.1)]
 # Create a list of parameters α,β

population = 10000
trial = 100
sd_value = 1.0

reward_1, reward_2 = 0.0, 3.0
 # if Reward 3 vs.0 conditions, reward_1, reward_2 = 0.0, 3.0
 # if Reward 5 vs.3 conditions, reward_1, reward_2 = 3.0, 5.0

t1 = time.time() 
dt_now = datetime.datetime.now()
print(dt_now)
 # Measure execution time

data_optimal_choice = np.zeros((len(parameter), trial)).tolist()
 # Create a list to store the number of people who chose the most profitable action.


for a in range(len(parameter)):
 # parameter loops : α,β = 0.1,0.1 , α,β = 0.1,0.2 , ... , α,β = 0.9,0.9 (9×9 = 81 times) 
    alpha = parameter[a][0]
    beta = parameter[a][1]
    
    pop_stpair = np.zeros((population, trial)).tolist()
     # Create a list to store all population's behaviour history (for 100trials)

    for b in range(population):
     # population loops : 10000 times

        Qmb_S0A1 = [0 for i in range(trial+1)]
        Qmb_S0A2 = [0 for i in range(trial+1)]
        Qmb_S1A1 = [0 for i in range(trial+1)]
        Qmb_S1A2 = [0 for i in range(trial+1)]
         # Create a list to store state-action pairs (for 100trials)
       
        action_record = [0 for i in range(trial)]
         # Create a list to store each individual's behaviour history (for 100trials)
        

        for c in range(trial): 
         # trials loops : 100 times
            s0_pA1 = 1/(1 + math.exp(-beta*(Qmb_S0A1[c]-Qmb_S0A2[c])))
            s0_pA2 = 1-s0_pA1            
            s0_choice = np.random.choice(["A1","A2"],p=[s0_pA1,s0_pA2])
            action_1 = "S0" + s0_choice
             # Starting from state S0, the first action selection is performed.


            if s0_choice == "A1":               
             # If action A1 is chosen, the transition is made to state S1 and a second action selection is performed.
                s1_pA1 = 1/(1 + math.exp(-beta*(Qmb_S1A1[c]-Qmb_S1A2[c])))
                s1_pA2 = 1-s1_pA1
                s1_choice = np.random.choice(["A1","A2"],p=[s1_pA1,s1_pA2])
                action_2 = "S1" + s1_choice


                if s1_choice == "A1":
                 # Different rewards are awarded depending on the second choice of action.
                 # Reward values are reversed on the boundary of 50 trials.
                    if c < (trial/2):
                        outcome = np.random.normal(reward_1, sd_value)
                    if c >= (trial/2):
                        outcome = np.random.normal(reward_2, sd_value)
                    
                    Qmb_S0A2[c+1] = Qmb_S0A2[c]
                    Qmb_S1A2[c+1] = Qmb_S1A2[c]
                    
                    Qmb_S1A1[c+1] = Qmb_S1A1[c]+alpha*(outcome-Qmb_S1A1[c])
                    Qmb_S0A1[c+1] = max(Qmb_S1A1[c+1],Qmb_S1A2[c+1])
                    action_record[c] = action_1 + action_2
                      

                if s1_choice == "A2":
                 # Different rewards are awarded depending on the second choice of action.
                 # Reward values are reversed on the boundary of 50 trials.               
                    if c < (trial/2):
                        outcome = np.random.normal(reward_2, sd_value)
                    if c >= (trial/2):
                        outcome = np.random.normal(reward_1, sd_value)
                    
                    Qmb_S0A2[c+1] = Qmb_S0A2[c]
                    Qmb_S1A1[c+1] = Qmb_S1A1[c]
                    
                    Qmb_S1A2[c+1] = Qmb_S1A2[c]+alpha*(outcome-Qmb_S1A2[c])
                    Qmb_S0A1[c+1] = max(Qmb_S1A1[c+1],Qmb_S1A2[c+1])
                    action_record[c] = action_1 + action_2


            if s0_choice == "A2":               
             # If action A2 is chosen for the first choice of action, the transition is made to state S2.
             # If the transition to state S2 is made, the second action selection is not performed.
                s2_action = "--"
                action_2 = "S2" + s2_action
                 
                outcome = np.random.normal(reward_1, sd_value)
                 
                Qmb_S0A1[c+1] = Qmb_S0A1[c]
                Qmb_S1A1[c+1] = Qmb_S1A1[c]
                Qmb_S1A2[c+1] = Qmb_S1A2[c]
                
                Qmb_S0A2[c+1] = Qmb_S0A2[c]+alpha*(outcome-Qmb_S0A2[c])              
                action_record[c] = action_1 + action_2


        pop_stpair[b] = action_record

       
    optimal_count1 = [[r[i] for r in pop_stpair].count("S0A1S1A2") for i in range(int(trial/2))]
    optimal_count2 = [[r[i] for r in pop_stpair].count("S0A1S1A1") for i in range(int(trial/2), trial, 1)]     
    data_optimal_choice[a] = optimal_count1 + optimal_count2
     # Count the number of people who chose the most profitable action.
 

t2 = time.time()
elapsed_time = t2-t1
print(f"elapsed time ： {elapsed_time}")
 # Display elapsed time


with open('Tree_MB_environmental_change.csv', 'w') as file:
 # Save data on the number of people who selected the most profitable action in a csv file.
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(data_optimal_choice)
