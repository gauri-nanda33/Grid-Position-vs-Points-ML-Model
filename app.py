import fastf1
import streamlit as st
import logging
import os
import pandas as pd
from fastf1.ergast import Ergast
ergast = Ergast()
logging.getLogger('fastf1').setLevel(logging.WARNING)
os.makedirs('my_cache', exist_ok=True)
fastf1.Cache.enable_cache('my_cache')

@st.cache_data
def load_data():

    drivers21 = ['VER', 'PER', 'HAM', 'BOT', 'RIC', 'NOR', 'VET', 'STR', 'ALO', 'OCO',
                'LEC', 'SAI', 'GAS', 'TSU', 'RAI', 'GIO', 'MAZ', 'MSC', 'LAT', 'RUS']

    drivers22 = ['VER', 'PER', 'LEC', 'SAI', 'RUS', 'HAM', 'NOR', 'RIC', 'ALO', 'OCO',
                'VET', 'STR', 'GAS', 'TSU', 'BOT', 'ZHO', 'MAG', 'MSC', 'LAT', 'DEV']

    drivers23 = ['VER', 'PER', 'ALO', 'HAM', 'RUS', 'SAI', 'LEC', 'NOR', 'STR', 'OCO',
                'GAS', 'TSU', 'BOT', 'ZHO', 'MAG', 'HUL', 'PIA', 'SAR', 'DEV', 'LAW', 'RIC']

    drivers24 = ['VER', 'NOR', 'LEC', 'PIA', 'SAI', 'RUS', 'HAM', 'PER', 'ALO', 'STR',
                'GAS', 'OCO', 'RIC', 'TSU', 'BOT', 'ZHO', 'MAG', 'HUL', 'ALB', 'SAR', 'COL','LAW','BEA']

    drivers25 = ['NOR', 'VER', 'PIA', 'LEC', 'RUS', 'ANT', 'HAM', 'SAI', 'ALO', 'STR',
                'GAS', 'OCO', 'TSU', 'LAW', 'BOT', 'BEA', 'MAG', 'HUL', 'ALB', 'HAD', 'COL']

    driv=[drivers21,drivers22,drivers23,drivers24,drivers25]
    years = [2021,2022,2023,2024,2025]

    records = []  # collect all rows, build DataFrame at the end
    standings=[]

    for j in range(len(years)):
        schedule = fastf1.get_event_schedule(years[j], include_testing=False)  # once per year

        for k in range(1, len(schedule) + 1):
            session = fastf1.get_session(years[j], k, "Qualifying")
            session.load(laps=False, telemetry=False,   # only load what you need
                        weather=False, messages=False)

            for i in driv[j]:
                try:
                    row = session.results[session.results['Abbreviation'] == i]
                    if row.empty:
                        continue  # driver didn't participate
                    pos = int(row['Position'].values[0])
                    records.append({
                        'driver':    i,
                        'year':      years[j],
                        'round':     k,
                        'quali_pos': pos
                    })
                except Exception:
                    continue  # handle any unexpected missing data
        yearstandings= ergast.get_driver_standings(season=years[j]).content[0]
        yearstandings['year'] = years[j]
        standings.append(yearstandings)

    df = pd.DataFrame(records)

    # Average qualifying position per driver per year
    summary = (df.groupby(['driver', 'year'])['quali_pos']
                .mean()
                .round(2)
                .reset_index()
                .rename(columns={'quali_pos': 'avg_quali_pos'}))

    #print(summary.sort_values('driver'))

    standings = pd.concat(standings, ignore_index=True)

    merged = summary.merge(
        standings[['driverCode', 'points', 'year']],
        left_on=['driver', 'year'],
        right_on=['driverCode', 'year']
    )
    #print(merged)
    #print(summary.sort_values('driver'))
    x_train = merged['avg_quali_pos'].to_numpy()  
    y_train = merged['points'].to_numpy()
    st.dataframe(merged)

    import math, copy
    import numpy as np
    import matplotlib.pyplot as plt

    # Load our data sets

    #Function to calculate the cost
    def compute_cost(x, y, w, b):

        m = x.shape[0]
        cost = 0

        for i in range(m):
            f_wb = w * x[i] + b
            cost = cost + (f_wb - y[i])**2
        total_cost = 1 / (2 * m) * cost

        return total_cost

    def compute_gradient(x, y, w, b):
        """
        Computes the gradient for linear regression
        Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters
        Returns
        dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b
        """

        # Number of training examples
        m = x.shape[0]
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            f_wb = w * x[i] + b
            dj_dw_i = (f_wb - y[i]) * x[i]
            dj_db_i = f_wb - y[i]
            dj_db += dj_db_i
            dj_dw += dj_dw_i
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db

    def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
        """
        Performs gradient descent to fit w,b. Updates w,b by taking
        num_iters gradient steps with learning rate alpha

        Args:
        x (ndarray (m,))  : Data, m examples
        y (ndarray (m,))  : target values
        w_in,b_in (scalar): initial values of model parameters
        alpha (float):     Learning rate
        num_iters (int):   number of iterations to run gradient descent
        cost_function:     function to call to produce cost
        gradient_function: function to call to produce gradient

        Returns:
        w (scalar): Updated value of parameter after running gradient descent
        b (scalar): Updated value of parameter after running gradient descent
        J_history (List): History of cost values
        p_history (list): History of parameters [w,b]
        """

        # An array to store cost J and w's at each iteration primarily for graphing later
        J_history = []
        p_history = []
        b = b_in
        w = w_in

        for i in range(num_iters):
            # Calculate the gradient and update the parameters using gradient_function
            dj_dw, dj_db = gradient_function(x, y, w , b)

            # Update Parameters using equation (3) above
            b = b - alpha * dj_db
            w = w - alpha * dj_dw

            # Save cost J at each iteration
            if i<100000:      # prevent resource exhaustion
                J_history.append( cost_function(x, y, w , b))
                p_history.append([w,b])
            # Print cost every at intervals 10 times or as many iterations if < 10

        return w, b, J_history, p_history #return w and J,w history for graphing

    # initialize parameters
    w_init = 0
    b_init = 0
    # gradient descent settings
    iterations = 100000
    tmp_alpha = 6.0e-4
    # run gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                        iterations, compute_cost, compute_gradient)
    print(f"(w,b) found by gradient descent: ({w_final:8.10f},{b_final:8.10f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_train, y_train, color='red', label='Actual data')

    x_line = np.linspace(x_train.min(), x_train.max(), 100)
    y_line = w_final * x_line + b_final

    ax.plot(x_line, y_line, color='blue', label='Model fit')
    ax.set_xlabel('Average qualifying position')
    ax.set_ylabel('Championship points')
    ax.set_title('F1 2021-2025 — Grid position vs Points')
    ax.legend()
    st.pyplot(fig)
    return merged, x_train, y_train, w_final, b_final

merged, x_train, y_train, w_final, b_final = load_data()