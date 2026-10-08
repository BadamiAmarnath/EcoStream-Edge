import pandas as pd  
from sklearn.ensemble import RandomForestRegressor  
import os  
import pickle  
df = pd.read_csv('data/carbon_grid_data.csv')  
X = df[['hour', 'day_of_week', 'month']]  
y = df['carbon_intensity']  
print("?? Training the predictive AI model...")  
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  
model.fit(X, y)  
os.makedirs('models', exist_ok=True)  
with open('models/grid_predictor.pkl', 'wb') as f: pickle.dump(model, f)  
print("? Success! Weights locked at models/grid_predictor.pkl") 
