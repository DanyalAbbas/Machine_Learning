import numpy as np
import pandas as pd
import math

df = pd.read_csv("test_scores.csv")

maths = df.math
cs = df.cs

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    learning_rate = 0.0003
    n = len(x)
    prev_cost = 0
    for i in range(iterations):
        y_pred = m_curr*x + b_curr
        cost = (1/n)*sum([i**2 for i in (y - y_pred)])
        md = -(2/n)*sum(x*(y - y_pred))
        bd = -(2/n)*sum(y - y_pred)
        m_curr = m_curr - learning_rate *md
        b_curr = b_curr - learning_rate *bd
        if math.isclose(cost, prev_cost, rel_tol = 1e-20):
            break
        prev_cost = cost
        print(f"m {m_curr}, b {b_curr}, cost {cost} iteration {i}")


gradient_descent(maths,cs)
