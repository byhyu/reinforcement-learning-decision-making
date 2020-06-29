from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
model = Sequential([
            layers.Dense(32, input_dim=4, activation='relu',kernel_initializer='he_uniform',bias_initializer='zeros'),  #, kernel_initializer='he_uniform'),
            layers.Dense(32, activation='relu',kernel_initializer='he_uniform',bias_initializer='zeros'),  #, kernel_initializer='he_uniform'),
            layers.Dense(2,activation='linear',kernel_initializer='he_uniform',bias_initializer='zeros') ##, name='q_value') 'random_uniform'
        ])
model.compile(optimizer=Adam(learning_rate=0.01),
             loss='mse')

import numpy as np
a = np.array([1,2,3,4])
b = np.reshape(a,[1,4])
# model.fit(np.random.random([10,4]),np.random.random([10,2]))
p = model.predict(b)
print(p)