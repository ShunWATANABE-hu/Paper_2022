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
lambda_value = 0.5

state_p1 = 0.5
state_p2 = 1-state_p1
 # Define the probability of each initial state being chosen.

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

    pop_action_1 = np.zeros((population, trial)).tolist()
    pop_action_2 = np.zeros((population, trial)).tolist()
     # Create a list to store all population's behaviour history (for 100trials)

    
    for b in range(population):
     # population loops : 10000 times

        Qmb_S1A1 = [0 for i in range(trial+1)]
        Qmb_S1A2 = [0 for i in range(trial+1)]
        Qmb_S2A3 = [0 for i in range(trial+1)]
        Qmb_S2A4 = [0 for i in range(trial+1)]
        Qmb_S3A5 = [0 for i in range(trial+1)]
        Qmb_S4A6 = [0 for i in range(trial+1)]
         # Create a list to store state-action pairs (for 100trials)
        
        action_1 = [0 for i in range(trial)]
        action_2 = [0 for i in range(trial)]
         # Create a list to store each individual's behaviour history (for 100trials)

        
        for c in range(trial):                
         # trials loops : 100 times

            state_1 = np.random.choice(["S1","S2"],p=[state_p1,state_p2])
             # One of the initial states is selected.

            if state_1 == "S1":
             # Starting from state S1, the first action selection is performed.
                s1_pA1 = 1/(1 + math.exp(-beta*(Qmb_S1A1[c]-Qmb_S1A2[c])))
                s1_pA2 = 1-s1_pA1            
                s1_choice = np.random.choice(["A1","A2"],p=[s1_pA1,s1_pA2])


                if s1_choice == "A1":
                 # Different rewards are awarded depending on the second choice of action.
                 # Reward values are reversed on the boundary of 50 trials.
                    Qmb_S1A1[c] = Qmb_S1A1[c] + alpha*(Qmb_S3A5[c]-Qmb_S1A1[c])
                    Qmb_S2A3[c] = Qmb_S2A3[c] + alpha*(Qmb_S3A5[c]-Qmb_S2A3[c])

                    if c < (trial/2):
                        outcome = np.random.normal(reward_1, sd_value)
                    if c >= (trial/2):
                        outcome = np.random.normal(reward_2, sd_value)

                    Qmb_S1A2[c+1] = Qmb_S1A2[c] 
                    Qmb_S2A4[c+1] = Qmb_S2A4[c]
                    Qmb_S4A6[c+1] = Qmb_S4A6[c]
                   
                    Qmb_S3A5[c+1] = Qmb_S3A5[c] + alpha*(outcome-Qmb_S3A5[c]) 
                    Qmb_S1A1[c+1] = Qmb_S1A1[c] + alpha*lambda_value*(outcome-Qmb_S1A1[c])
                    Qmb_S2A3[c+1] = Qmb_S2A3[c] + alpha*lambda_value*(outcome-Qmb_S2A3[c])

                    action_1[c] = s1_choice
                    action_2[c] = "A5"
                  
                if s1_choice == "A2":
                 # Different rewards are awarded depending on the second choice of action.
                 # Reward values are reversed on the boundary of 50 trials.
                    Qmb_S1A2[c] = Qmb_S1A2[c] + alpha*(Qmb_S4A6[c]-Qmb_S1A2[c])
                    Qmb_S2A4[c] = Qmb_S2A4[c] + alpha*(Qmb_S4A6[c]-Qmb_S2A4[c])

                    if c < (trial/2):
                        outcome = np.random.normal(reward_2, sd_value)
                    if c >= (trial/2):
                        outcome = np.random.normal(reward_1, sd_value)

                    Qmb_S1A1[c+1] = Qmb_S1A1[c] 
                    Qmb_S2A3[c+1] = Qmb_S2A3[c]
                    Qmb_S3A5[c+1] = Qmb_S3A5[c]
                   
                    Qmb_S4A6[c+1] = Qmb_S4A6[c] + alpha*(outcome-Qmb_S4A6[c]) 
                    Qmb_S1A2[c+1] = Qmb_S1A2[c] + alpha*lambda_value*(outcome-Qmb_S1A2[c])
                    Qmb_S2A4[c+1] = Qmb_S2A4[c] + alpha*lambda_value*(outcome-Qmb_S2A4[c])

                    action_1[c] = s1_choice
                    action_2[c] = "A6"

                
            if state_1 == "S2":
             # Starting from state S2, the first action selection is performed.
                s2_pA3 = 1/(1 + math.exp(-beta*(Qmb_S2A3[c]-Qmb_S2A4[c])))
                s2_pA4 = 1-s2_pA3            
                s2_choice = np.random.choice(["A3","A4"],p=[s2_pA3,s2_pA4])

               
                if s2_choice == "A3":
                 # Different rewards are awarded depending on the second choice of action.
                 # Reward values are reversed on the boundary of 50 trials.
                    Qmb_S2A3[c] = Qmb_S2A3[c] + alpha*(Qmb_S3A5[c]-Qmb_S2A3[c])
                    Qmb_S1A1[c] = Qmb_S1A1[c] + alpha*(Qmb_S3A5[c]-Qmb_S1A1[c])

                    if c < (trial/2):
                        outcome = np.random.normal(reward_1, sd_value)
                    if c >= (trial/2):
                        outcome = np.random.normal(reward_2, sd_value)

                    Qmb_S1A2[c+1] = Qmb_S1A2[c] 
                    Qmb_S2A4[c+1] = Qmb_S2A4[c]
                    Qmb_S4A6[c+1] = Qmb_S4A6[c]
                   
                    Qmb_S3A5[c+1] = Qmb_S3A5[c] + alpha*(outcome-Qmb_S3A5[c]) 
                    Qmb_S2A3[c+1] = Qmb_S2A3[c] + alpha*lambda_value*(outcome-Qmb_S2A3[c])
                    Qmb_S1A1[c+1] = Qmb_S1A1[c] + alpha*lambda_value*(outcome-Qmb_S1A1[c])

                    action_1[c] = s2_choice
                    action_2[c] = "A5"

                    
                if s2_choice == "A4":
                 # Different rewards are awarded depending on the second choice of action.
                 # Reward values are reversed on the boundary of 50 trials.
                    Qmb_S2A4[c] = Qmb_S2A4[c] + alpha*(Qmb_S4A6[c]-Qmb_S2A4[c])
                    Qmb_S1A2[c] = Qmb_S1A2[c] + alpha*(Qmb_S4A6[c]-Qmb_S1A2[c])

                    if c < (trial/2):
                        outcome = np.random.normal(reward_2, sd_value)
                    if c >= (trial/2):
                        outcome = np.random.normal(reward_1, sd_value)

                    Qmb_S1A1[c+1] = Qmb_S1A1[c] 
                    Qmb_S2A3[c+1] = Qmb_S2A3[c]
                    Qmb_S3A5[c+1] = Qmb_S3A5[c]
                   
                    Qmb_S4A6[c+1] = Qmb_S4A6[c] + alpha*(outcome-Qmb_S4A6[c]) 
                    Qmb_S2A4[c+1] = Qmb_S2A4[c] + alpha*lambda_value*(outcome-Qmb_S2A4[c])
                    Qmb_S1A2[c+1] = Qmb_S1A2[c] + alpha*lambda_value*(outcome-Qmb_S1A2[c])

                    action_1[c] = s2_choice
                    action_2[c] = "A6"


        pop_action_1[b] = action_1
        pop_action_2[b] = action_2
        

    optimal_50 = [[r[i] for r in pop_action_2].count("A5") for i in range(int(trial/2))]
    optimal_100 = [[r[i] for r in pop_action_2].count("A6") for i in range(int(trial/2), trial, 1)]
    data_optimal_choice[a] = optimal_50 + optimal_100
     # Count the number of people who chose the most profitable action.

    
t2 = time.time()
elapsed_time = t2-t1
print(f"elapsed time ： {elapsed_time}")
 # Display elapsed time


with open('Rocket_MB_environmental_change.csv', 'w') as file:
 # Save data on the number of people who selected the most profitable action in a csv file.
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(data_optimal_choice)

