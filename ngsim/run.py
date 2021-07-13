import pandas as pd

dataset_path = 'dataset/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv'
df = pd.read_csv(dataset_path)
print(df.head())