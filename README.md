# Crypto currency predictive model

** run the project With tensorflow 1.2.1

Steps to run:
1) Download the demo dataset from : https://drive.google.com/file/d/1wHJdFBJO0EYCq_rO5VSjDws4_d69Jvyz/view?usp=sharing
   and put the file under ./datasets/crypto_hist
2) run the python script ./optimizer.py
3) After running the optimization, the result will stored in ./models_data/[df_tf]

The code of the model is mainly from the ./models/lstm_model.py

The result contains the best models, the optimizing performance csv log [hyper_param_output.csv], and the SKOPT optimizing history [skopt_lstm.pkl]
Note that: backtest function is still under developing and it is not avalible yet