#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd

original_df = pd.read_csv('Obfuscated-MalMem2022.csv', sep=',', encoding='utf-8')
original_df.shape


# In[3]:


original_df.describe()


# In[4]:


original_df.tail(150)


# In[5]:


df = original_df.copy()
df.shape


# In[6]:


import pandas as pd

def select_every_nth_row(df, n=1000):
   
    selected_rows = df.iloc[::n]
    return selected_rows

# Example usage:
# Assuming df is your DataFrame with shape (58596, 57)
selected_rows = select_every_nth_row(df, n=1000)


# In[7]:


# Check if all values in each column are identical
for column in df.columns:
    if df[column].nunique() == 1:
        print(f"All values in {column} are identical.")


# In[8]:


# drop identical columns
# List of columns to drop
columns_to_drop = ['pslist.nprocs64bit', 'handles.nport', 'svcscan.interactive_process_services']

# Drop the specified columns
df.drop(columns=columns_to_drop, inplace=True)


# In[9]:


import pandas as pd
from scipy.stats import zscore

# Print the number of missing values
print("Number of Missing Values:")
print(df.isnull().sum())

# Print the number of duplicate rows
print("\nNumber of Duplicate Rows:", df.duplicated().sum())

# Handling Missing Values
df.fillna(method="ffill", inplace=True)  # Forward fill missing values

# Removing Duplicates
df.drop_duplicates(inplace=True)

# Data Type Conversion
df["Class"] = df["Class"].astype("category")

#Outlier Handling (Example: Using Z-score)
# z_scores = zscore(df.select_dtypes(include=['int64', 'float64']))
# df = df[(z_scores < 3).all(axis=1)]

# Handling Categorical Data (One-Hot Encoding)
df = pd.get_dummies(df, columns=["Class"], drop_first=True)

# Feature Scaling (Min-Max Scaling)
# df[df.select_dtypes(include=['int64', 'float64']).columns] = \
#     (df[df.select_dtypes(include=['int64', 'float64']).columns] - df[df.select_dtypes(include=['int64', 'float64']).columns].min()) / \
#     (df[df.select_dtypes(include=['int64', 'float64']).columns].max() - df[df.select_dtypes(include=['int64', 'float64']).columns].min())

# Removing Unnecessary Columns
# columns_to_remove = ["Column1", "Column2"]  # List of columns to remove
# df.drop(columns=columns_to_remove, inplace=True)


# In[10]:


# Data Sanity Check
print("\nData Info:")
print(df.info())
print("\nFirst Few Rows:")
print(df.head())


# In[11]:


print("DataFrame Shape:", df.shape)
print("Class Distribution:", df["Class_Malware"].value_counts())


# In[12]:


# Print the number of missing values
print("Number of Missing Values:")
print(df.isnull().sum())


# In[13]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Separate features and target
y = df["Class_Malware"]
X = df.drop(columns=["Category", "Class_Malware"])


# In[14]:


# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Standardize your features (recommended before PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[15]:


# Apply PCA for feature extraction
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance ratios
explained_variance_ratio = pca.explained_variance_ratio_

# Plot cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
plt.plot(cumulative_variance_ratio)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.show()


# In[16]:


# Determine the number of components to retain (e.g., capturing 95% of variance)
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components to retain for 95% variance: {n_components}")

# Now you can perform PCA with the selected number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)


# In[17]:


# Visualize the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratios:", explained_variance_ratio)


# In[18]:


# Plot explained variance ratios for the first 10 components
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for the First 10 Principal Components')
plt.grid(True)
plt.show()


# In[19]:


from sklearn.manifold import TSNE

# Feature extraction using t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
X_tsne = tsne.fit_transform(X_scaled)


# In[20]:


print(X_tsne)


# In[21]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# # Feature extraction using LDA
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)


# In[22]:


X_lda


# In[23]:


print(df["Class_Malware"].unique())


# In[27]:


import matplotlib.pyplot as plt

# Create subplots for side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot PCA
axes[0].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label="Benign", color="blue", alpha=0.5)
axes[0].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label="Malicious", color="red", alpha=0.5)
axes[0].set_title("PCA")
axes[0].legend()

# Plot LDA
axes[2].scatter(X_lda[y == 0, 0], X_lda[y == 0, 0], label="Benign", color="blue", alpha=0.5)
axes[2].scatter(X_lda[y == 1, 0], X_lda[y == 1, 0], label="Malicious", color="red", alpha=0.5)
axes[2].set_title("LDA")
axes[2].legend()

plt.show()


# In[28]:


# Import necessary libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# XGBoost for Feature Extraction
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Use the trained XGBoost model to get feature importances
feature_importances = xgb_model.feature_importances_

# Select the top N important features
N = 14  # Change this value based on the number of features you want to select
selected_feature_indices = np.argsort(feature_importances)[::-1][:N]
selected_features = X.columns[selected_feature_indices]

# Subset the dataset with the selected features
X_train_selected = X_train.iloc[:, selected_feature_indices]
X_test_selected = X_test.iloc[:, selected_feature_indices]

# Normalize the data
scaler = StandardScaler()
X_train_selected = scaler.fit_transform(X_train_selected)
X_test_selected = scaler.transform(X_test_selected)


# In[29]:


# Sort and visualize feature importances
N = 14  # Number of top features to select
sorted_indices = np.argsort(feature_importances)[::-1]
selected_features = X.columns[sorted_indices[:N]]

plt.figure(figsize=(10, 6))
plt.bar(selected_features, feature_importances[sorted_indices[:N]])
plt.title('Top Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()


# In[30]:


selected_features


# In[31]:


import numpy as np
import pandas as pd
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Function to randomly select data from a CSV file
def random_data_from_csv(file_path, features, num_samples=1000):
    data = pd.read_csv(file_path)
    selected_data = data[features].sample(n=num_samples)
    return selected_data.values

# Define function for training the DQN agent
def train_dqn(agent, data, episodes=1000, batch_size=32):
    scores = []
    for e in range(episodes):
        state = data[np.random.randint(0, len(data))]  # Sample state
        state = np.reshape(state, [1, agent.state_size])
        done = False
        score = 0
        while not done:
            action = agent.act(state)
            next_state = data[np.random.randint(0, len(data))]  # Sample next state
            next_state = np.reshape(next_state, [1, agent.state_size])
            reward = np.random.randn()  # Sample reward
            done = np.random.choice([True, False])  # Sample done flag
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        scores.append(score)
        print("Episode:", e+1, "  Score:", score)
    return scores

# Example usage
if __name__ == "__main__":
    # Initialize environment
    state_size = 14  # Number of selected features
    action_size = 2  # Example action size
    agent = DQNAgent(state_size, action_size)
    
    # Example of using random_data_from_csv function
    features = ['svcscan.nservices', 'svcscan.process_services',
                'handles.avg_handles_per_proc', 'handles.ndesktop',
                'callbacks.ncallbacks', 'malfind.commitCharge',
                'pslist.nproc', 'psxview.not_in_deskthrd',
                'handles.nevent', 'pslist.avg_handlers',
                'psxview.not_in_session_false_avg', 'malfind.protection',
                'handles.nkey', 'ldrmodules.not_in_load']
    selected_data = random_data_from_csv("Data Normalised.csv", features, num_samples=1000)
    
    # Train the DQN agent
    scores = train_dqn(agent, selected_data)

    # Plot the scores
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training Performance')
    plt.show()


# In[33]:


# Assuming you have a test dataset and labels for evaluation
# Here, I assume `X_test` contains the test data and `y_test` contains the corresponding labels
X_test = random_data_from_csv("Obfuscated-MalMem2022.csv", features, num_samples=100)  # Assuming you have a function to load test data
y_test = np.random.randint(2, size=len(X_test))  # Example labels for evaluation

# Predict labels using the trained DQN agent
y_pred = []
for state in X_test:
    state = np.reshape(state, [1, agent.state_size])
    action = agent.act(state)
    y_pred.append(action)
y_pred = np.array(y_pred)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Compute classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


# In[ ]:




