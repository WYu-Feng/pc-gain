## Use this for tf 2.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import tensorflow as tf
import numpy as np
from utils import normalization, renormalization , rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
from tqdm import tqdm

from cluster import KM , SC , SB , KMPP, AC
import sklearn.svm as svm
from sklearn import preprocessing


def PC_GAIN (incomplete_data_x , gain_parameters , data_m):
    '''Impute missing values in incomplete_data_x

    Args:
    - incomplete_data_x: original data with missing values
    - gain_parameters: PC_GAIN network parameters:
    - batch_size: Batch size，64
    - hint_rate: Hint rate，0.9
    - alpha: Hyperparameter，200
    - beta: Hyperparameter，20
    - lambda_: Hyperparameter，0.2
    - k: Hyperparameter，4
    - iterations: Iterations，10000

    Returns:
    - imputed_data: imputed data
    '''
    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    beta = gain_parameters['beta']
    lambda_ = gain_parameters['lambda_']
    k = gain_parameters['k']
    iterations = gain_parameters['iterations']
    cluster_species = gain_parameters['cluster_species']
    
    # Other parameters
    no, dim = incomplete_data_x.shape
    # Hidden state dimensions
    h_dim = int(dim)
    # Normalization
    norm_data , norm_parameters = normalization(incomplete_data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)


    ## PC_GAIN architecture   
    X = tf.placeholder(tf.float32, shape = [None, dim])
    M = tf.placeholder(tf.float32, shape = [None, dim])
    H = tf.placeholder(tf.float32, shape = [None, dim])

    Z = tf.placeholder(tf.float32, shape = [None, dim])
    Y = tf.placeholder(tf.float32, shape = [None, k])
    
    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    #Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
    G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape = [dim]))
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    C_W1 = tf.Variable(xavier_init([dim, h_dim]))
    C_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    C_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    C_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    C_W3 = tf.Variable(xavier_init([h_dim, k]))
    C_b3 = tf.Variable(tf.zeros(shape = [k]))  # 分类器
    theta_C = [C_W1, C_b1, C_W2, C_b2, C_W3, C_b3]
  
    ## PC_GAIN functions
    # Generator
    def generator(x,m):
        # Concatenate Mask and Data
        inputs = tf.concat(values = [x, m], axis = 1) 
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
        return G_prob
      
    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values = [x, h], axis = 1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit
    
    # Classer (neural network classifier mentioned in the paper)
    def classer(feature):
        C_h1 = tf.nn.relu(tf.matmul(feature, C_W1) + C_b1)
        C_h2 = tf.nn.relu(tf.matmul(C_h1, C_W2) + C_b2)
        C_h3 = tf.matmul(C_h2, C_W3) + C_b3
        C_prob = tf.nn.softmax(C_h3)
        return C_prob  
  
    ## PC_GAIN structure
    # Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1-M)

    # Discriminator
    D_prob, D_logit = discriminator(Hat_X, H)
    
    ## PC_GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8))
    G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
    G_loss_with_C = -tf.reduce_mean(Y * tf.log(Y + 1e-8))
    MSE_loss = tf.reduce_mean((M * X - M * G_sample) * (M * X - M * G_sample)) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss_pre = G_loss_temp + alpha * MSE_loss
    G_loss = G_loss_temp + alpha * MSE_loss + beta * G_loss_with_C
    
    ## PC_GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver_pre = tf.train.AdamOptimizer().minimize(G_loss_pre, var_list=theta_G)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    
    ## Iterations
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        ##Select pre-training data
        loss_rate = []
        for i in range(no):
            index = 0
            for j in range(dim):
                if data_m[i,j] == 0:
                    index = index + 1
            loss_rate.append([index , i])
        loss_rate = sorted(loss_rate,key=(lambda x:x[0]))
        no_x_L = int(no * lambda_) 
        index_x_L = []
        for i in range(no_x_L):
            index_x_L.append(loss_rate[i][1])
        norm_data_x_L = norm_data_x[index_x_L, :]
        data_m_L = data_m[index_x_L, :]
        
        ##Pre-training
        print('...Pre-training')
        for it in tqdm(range(int(iterations * 0.7))):
            batch_idx = sample_batch_index(no_x_L, batch_size)
            X_mb = norm_data_x_L[batch_idx, :]
            M_mb = data_m_L[batch_idx, :]
            Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
            H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
            H_mb = M_mb * H_mb_temp
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
            _, D_loss_curr, D_logit_curr, D_prob_curr = sess.run([D_solver, D_loss_temp, D_logit, D_prob], feed_dict = {M: M_mb, X: X_mb, H:H_mb})
            _, G_loss_curr, MSE_loss_curr = sess.run([G_solver_pre, G_loss_temp, MSE_loss], feed_dict = {X: X_mb, M: M_mb, H:H_mb})
            
        Z_mb = uniform_sampler(0, 0.01, no_x_L, dim) 
        M_mb = data_m_L
        X_mb = norm_data_x_L
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
        imputed_data_L = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
        imputed_data_L = data_m_L * norm_data_x_L + (1 - data_m_L) * imputed_data_L
        
        ## Select different clustering methods
        if cluster_species == 'KM':
            data_c , data_class = KM(imputed_data_L, k)
        elif cluster_species == 'SC':
            data_c , data_class = SC(imputed_data_L, k)
        elif cluster_species == 'AC':
            data_c , data_class = AC(imputed_data_L, k)
        elif cluster_species == 'KMPP': 
            data_c , data_class = KMPP(imputed_data_L, k)         
        else:
            exit('have not this cluster methods')
        
        ## Pseudo-label training multi-classification SVM
        ## You can also choose other classifiers, 
        ## such as the neural network classifier mentioned in the paper
        coder = preprocessing.OneHotEncoder()
        model = svm.SVC(kernel="linear", decision_function_shape="ovo")   
        coder.fit(data_class.reshape(-1,1))
        model.fit(imputed_data_L, data_class)
        
        ## Updata the generator G and the discriminator D
        ## To avoid the effects of pre-training, 
        ## you can also choose to reinitialize the generator parameters
        for it in tqdm(range(iterations)):
            batch_idx = sample_batch_index(no, batch_size)
            X_mb = norm_data_x[batch_idx, :]
            M_mb = data_m[batch_idx, :]
            Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
            H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
            H_mb = M_mb * H_mb_temp
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
            _, D_loss_curr, D_logit_curr, D_prob_curr = sess.run([D_solver, D_loss_temp, D_logit, D_prob], feed_dict = {M: M_mb, X: X_mb, H:H_mb})
            
            ## Introducing pseudo label supervision
            Hat_X_curr = sess.run(Hat_X, feed_dict = {X: X_mb, M: M_mb, H:H_mb})
            y_pred = model.predict(Hat_X_curr)
            sample_prob = coder.transform(y_pred.reshape(-1,1)).toarray()  
            
            _, G_loss_curr, MSE_loss_curr , G_loss_with_C_curr = sess.run([G_solver, G_loss_temp, MSE_loss, G_loss_with_C], feed_dict = {X: X_mb, M: M_mb, H:H_mb , Y:sample_prob})
            
        ## Return imputed data 
        Z_mb = uniform_sampler(0, 0.01, no, dim) 
        M_mb = data_m
        X_mb = norm_data_x          
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
        imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
        imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
        imputed_data = renormalization(imputed_data, norm_parameters)
    return imputed_data

    
    
      