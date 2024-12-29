

# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
import pandas as pd
import scipy as sc
from scipy.special import digamma, gamma
import itertools
import copy

from mpmath import *
from sympy import *
#from sympy.printing.theanocode import theano_function
from sympy.utilities.autowrap import ufuncify

from pysymbolic.models.special_functions import *
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from tqdm import tqdm, trange, tqdm_notebook, tnrange

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sympy import Integral, Symbol
from sympy.abc import x, y
from scipy import special

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def is_ipython():
    
    try:
        
        __IPYTHON__
        
        return True
    
    except NameError:
        
        return False


def basis(a, b, c, x_val):
    """Numerically stable basis function calculation"""
    epsilon = 0.001
    x_val = np.clip(x_val + epsilon, -1e10, 1e10)
    
    # Protect against invalid parameter values
    a = np.clip(float(a), 0.1, 10.0)
    b = np.clip(float(b), 0.1, 10.0)
    c = np.clip(float(c), 0.1, 10.0)
    
    cx = np.clip(c * x_val, -1e10, 1e10)
    cx_power = np.clip(np.power(cx, a), -1e10, 1e10)
    
    # Calculate digamma and gamma values safely
    indices = np.arange(1, 5)
    ab_terms = a - b + indices
    digamma_terms = special.digamma(np.clip(ab_terms, 0.1, 30.0))
    gamma_terms = special.gamma(np.clip(ab_terms, 0.1, 30.0))
    
    # Compute the basis terms
    log_cx = np.log(np.abs(cx) + epsilon)
    terms = []
    
    for i, (d, g) in enumerate(zip(digamma_terms, gamma_terms)):
        power = min(i, 3)
        coef = 6 if power == 3 else 2 if power == 2 else 1
        term = coef * np.power(cx, power) * (d - log_cx) / g
        terms.append(term)
    
    result = cx_power * np.sum(terms, axis=0)
    return np.clip(result, -1e10, 1e10)

def basis_expression(a, b, c, hyper_order=[1, 2, 2, 2]):
    """Generate symbolic expression for basis function"""
    x = symbols('x')
    
    # Calculate terms for the series expansion
    terms = []
    for i in range(1, 5):
        coef = 6 if i == 4 else 2 if i == 3 else 1
        term = coef * (c * x)**(i-1) / special.gamma(a - b + i)
        terms.append(term)
    
    # Combine terms
    expression = (c * x)**a * sum(terms)
    return expression
    

def basis_grad(a, b, c, x):
    """Calculate gradients for basis function parameters"""
    epsilon = 0.001
    x = x + epsilon
    
    K = [special.digamma(a - b + i) for i in range(1, 5)]
    G = [special.gamma(a - b + i) for i in range(1, 5)]
    cx = c * x
    log_cx = np.log(cx)
    
    # Gradient with respect to a
    nema = [
        6 * (cx**3) * (K[3] - log_cx) / G[3],
        2 * (cx**2) * (-K[2] + log_cx) / G[2],
        cx * (K[1] - log_cx) / G[1],
        -1 * (K[0] - log_cx) / G[0]
    ]
    grad_a = (cx**a) * sum(nema)
    
    # Gradient with respect to b
    nemb = [
        -6 * (cx**3) * K[3] / G[3],
        2 * (cx**2) * K[2] / G[2],
        -1 * cx * K[1] / G[1],
        K[0] / G[0]
    ]
    grad_b = (cx**a) * sum(nemb)
    
    # Gradient with respect to c
    nemc = [
        -1 * (c**2) * (x**3) * (6 * a + 18) / G[3],
        (c * (x**2)) * (4 + 2 * a) / G[2],
        -1 * x * (1 + a) / G[1],
        a / (c * G[0])
    ]
    grad_c = (cx**a) * sum(nemc)
    
    return np.clip(grad_a, -1e10, 1e10), np.clip(grad_b, -1e10, 1e10), np.clip(grad_c, -1e10, 1e10)



def tune_single_dim(lr, n_iter, x, y, verbosity=False):
    
    epsilon   = 0.001
    x         = x + epsilon
    
    a         = 2
    b         = 1
    c         = 1
    
    batch_size  = np.min((x.shape[0], 500)) 
    
    for u in range(n_iter):
        
        batch_index = np.random.choice(list(range(x.shape[0])), size=batch_size)
        
        new_grads   = basis_grad(a, b, c, x[batch_index])
        func_true   = basis(a, b, c, x[batch_index])
        
        loss        =  np.mean((func_true - y[batch_index])**2)
        
        if verbosity:
        
            print("Iteration: %d \t--- Loss: %.3f" % (u, loss))
        
        grads_a   = np.mean(2 * new_grads[0] * (func_true - y[batch_index]))
        grads_b   = np.mean(2 * new_grads[1] * (func_true - y[batch_index]))
        grads_c   = np.mean(2 * new_grads[2] * (func_true - y[batch_index]))
        
        a         = a - lr * grads_a
        b         = b - lr * grads_b
        c         = c - lr * grads_c
        
    return a, b, c 


def compose_features(params, X):
    
    X_out = [basis(a=float(params[k, 0]), b=float(params[k, 1]), c=float(params[k, 2]), 
                   x=X[:, k], hyper_order=[1, 2, 2, 2]) for k in range(X.shape[1])] 
    
    return np.array(X_out).T
    

class symbolic_metamodel:
    
    def __init__(self, model, X, mode="classification", n_jobs=-1):
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        
        self.feature_expander = PolynomialFeatures(2, include_bias=False, 
                                                  interaction_only=True)
        self.X = X
        self.X_new = self.feature_expander.fit_transform(X)
        self.X_names = self.feature_expander.get_feature_names_out()
        
        if mode == "classification":
            proba = np.clip(model.predict_proba(X)[:, 1], 1e-10, 1-1e-10)
            self.Y_r = np.log(proba / (1 - proba))
        else:
            self.Y_r = model.predict(X)
        
        self.num_basis = self.X_new.shape[1]
        self.params = np.tile([1.39, 1.02, 1.49], [self.num_basis, 1])
        
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            
            self.tqdm_mode = tqdm_notebook
            
        else:
            
            self.tqdm_mode = tqdm
    
    def _tune_single_dim(self, dim_idx, lr=0.1, n_iter=500):
        """Tune parameters for a single dimension"""
        x = self.X_new[:, dim_idx]
        y = self.Y_r
        batch_size = min(len(x), 500)
        
        a, b, c = 2.0, 1.0, 1.0
        best_params = [a, b, c]
        best_loss = float('inf')
        
        for _ in range(n_iter):
            batch_idx = np.random.choice(len(x), batch_size)
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]
            
            func_val = basis(a, b, c, x_batch)
            loss = np.mean((func_val - y_batch)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_params = [a, b, c]
            
            grad_a, grad_b, grad_c = basis_grad(a, b, c, x_batch)
            
            # Scale gradients and apply updates
            grad_scale = np.mean(np.abs([grad_a, grad_b, grad_c]))
            if grad_scale > 1e-8:
                lr_scaled = lr / grad_scale
                
                a = np.clip(a - lr_scaled * np.mean(grad_a * (func_val - y_batch)), 0.1, 10.0)
                b = np.clip(b - lr_scaled * np.mean(grad_b * (func_val - y_batch)), 0.1, 10.0)
                c = np.clip(c - lr_scaled * np.mean(grad_c * (func_val - y_batch)), 0.1, 10.0)
        
        return best_params
    
    def set_equation(self, reset_init_model=False):
         
        self.X_init           = compose_features(self.params, self.X_new)
        
        if reset_init_model:
            
            self.init_model   = Ridge(alpha=.1, fit_intercept=False) #LinearRegression
            
            self.init_model.fit(self.X_init, self.Y_r)
    
    def fit(self, num_iter=10, batch_size=100, learning_rate=0.01):
        print("Tuning basis functions in parallel...")
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self._tune_single_dim, i)
                for i in range(self.X.shape[1])
            ]
            tuned_params = np.array([f.result() for f in futures])
        
        self.params[:self.X.shape[1]] = tuned_params
        
        # Initialize model with tuned parameters
        X_init = np.array([
            basis(self.params[j, 0], self.params[j, 1], self.params[j, 2], self.X_new[:, j])
            for j in range(self.num_basis)
        ]).T
        
        self.init_model = Ridge(alpha=0.1, fit_intercept=False)
        self.init_model.fit(X_init, self.Y_r)
        
        print("Optimizing metamodel...")
        best_loss = float('inf')
        patience = 5
        no_improve = 0
        
        for i in range(num_iter):
            batch_idx = np.random.choice(len(self.X_new), batch_size)
            X_batch = np.array([
                basis(self.params[j, 0], self.params[j, 1], self.params[j, 2], self.X_new[batch_idx, j])
                for j in range(self.num_basis)
            ]).T
            y_batch = self.Y_r[batch_idx]
            
            pred = self.init_model.predict(X_batch)
            loss = np.mean((y_batch - pred)**2)
            
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                learning_rate *= 0.5
                no_improve = 0
                if learning_rate < 1e-6:
                    print("Early stopping due to learning rate decay")
                    break
            
            grad = 2 * (pred - y_batch)[:, None] * X_batch
            grad_mean = np.mean(grad, axis=0)
            grad_scale = np.mean(np.abs(grad_mean))
            
            if grad_scale > 1e-8:
                self.init_model.coef_ -= learning_rate * grad_mean / grad_scale
            
            if i % 5 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}, LR: {learning_rate:.6f}")

    def evaluate(self, X):
        X_modified = self.feature_expander.transform(X)
        X_modified_ = np.array([
            basis(self.params[j, 0], self.params[j, 1], self.params[j, 2], X_modified[:, j])
            for j in range(self.num_basis)
        ]).T
        Y_pred_r = self.init_model.predict(X_modified_)
        return 1 / (1 + np.exp(-Y_pred_r)) 
    
    def symbolic_expression(self):
    
        dims_ = []

        for u in range(self.num_basis):

            new_symb = self.X_names[u].split(" ")

            if len(new_symb) > 1:
    
                S1 = Symbol(new_symb[0].replace("x", "X"), real=True)
                S2 = Symbol(new_symb[1].replace("x", "X"), real=True)
        
                dims_.append(S1 * S2)
    
            else:
        
                S1 = Symbol(new_symb[0].replace("x", "X"), real=True)
    
                dims_.append(S1)
        
        self.dim_symbols = dims_
        
        sym_exact   = 0
        sym_approx  = 0
        x           = symbols('x')

        for v in range(self.num_basis):
    
            f_curr      = basis_expression(a=float(self.params[v,0]), 
                                           b=float(self.params[v,1]), 
                                           c=float(self.params[v,2]))
        
            sym_exact  += sympify(str(self.init_model.coef_[v] * re(f_curr.expression()))).subs(x, dims_[v])
            sym_approx += sympify(str(self.init_model.coef_[v] * re(f_curr.approx_expression()))).subs(x, dims_[v])    
        
        return 1/(1 + exp(-1*sym_exact)), 1/(1 + exp(-1*sym_approx))   
    
    
    def get_gradient_expression(self):
        """Generate symbolic gradient expressions"""
        # Get only the input dimensions (not interaction terms)
        
        # Get the exact expression
        exact_expr, _ = self.symbolic_expression()
        
        diff_dims = self.dim_symbols[:self.X.shape[1]]
        # Calculate gradients symbolically
        gradients_ = [diff(exact_expr, diff_dims[k]) for k in range(len(diff_dims))]
        
        # Convert dimension symbols to strings for mapping
        diff_dims = [str(diff_dims[k]) for k in range(len(diff_dims))]
        
        # Create lambda functions for evaluation
        evaluator = [lambdify(diff_dims, gradients_[k], modules=['math']) 
                    for k in range(len(gradients_))]
        
        return gradients_, diff_dims, evaluator
    

    def _gradient(self, gradient_expressions, diff_dims, evaluator, x_in):
        """Evaluate gradient expressions at a point"""
        # Create dictionary mapping symbols to values
        Dict_syms = dict(zip(diff_dims, x_in))
        
        # Evaluate each gradient component
        grad_out = [abs(evaluator[k](**Dict_syms)) for k in range(len(evaluator))]
        
        return np.array(grad_out)
    
    def get_instancewise_scores(self, X_in):
        """Calculate gradients for each instance"""
        gr_exp, diff_dims, evaluator = self.get_gradient_expression()
        
        # Calculate gradients for each input point
        grads_ = [self._gradient(gr_exp, diff_dims, evaluator, X_in[k, :]) 
                 for k in range(X_in.shape[0])]
        
        return grads_
    
        
