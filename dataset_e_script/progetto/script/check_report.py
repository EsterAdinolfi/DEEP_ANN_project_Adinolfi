import pandas as pd
path = r'c:\Users\eadin\Desktop\DEEP_ANN_project_Adinolfi\dataset_e_script\progetto\risultati\pythia_160m\report_topic_pythia_160m.csv'
df = pd.read_csv(path)
print(df[['most_disruptive_threat','most_effective_threat_validity','most_effective_threat_consistency','avg_validity','consistency_score']].head())
print('counts disruptive:')
print(df['most_disruptive_threat'].value_counts() / len(df) * 100)
