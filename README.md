# ReLU Nonlinear Matrix Decomposition (ReLU-NMD)

This MATLAB codes provides algorithms to solve the following Nonlinear Matrix Decomposition (NMD) problems with the ReLU function:  
Given a nonnegative matrix X in R^{m x n} and an integer r, solve  

        min_{W in R^{m x r},H in R^{r x n}} ||X - max(0,W*H)||_F^2.  
        
This problem, which we refer to as *ReLU-NMD*, was introduced in a paper by Saul [S22]. 

The algorithms implemented are the following: 
- A-NMD and 3B-NMD which are the two most effective algorithms for ReLU-NMD, according to our experiments, and presented in our paper [S+23]. 
- The algorithms of [S22], A-Naive and EM, and of their accelerated versions from the follow-up paper [S23], A-Naive-NMD and A-EM. 

You can run the file RunMe.me to have a run a simple example comparing A-NMD and 3B-NMD on a synthetic data set. 

All experiments from our paper can be found in the folder "numerical experiments". 

See our paper [S+23] for more details. 


## References 
[S22] L.K. Saul, *A nonlinear matrix decomposition for mining the zeros of sparse data*, SIAM Journal on Mathematics of Data Science 4(2), 431-463, 2022.  
[S23] L.K. Saul, *A geometrical connection between sparse and low-rank matrices and its application to manifold learning*, Transactions on Machine Learning Research, 2023.  
[S+23] G. Seraghiti, A. Awari, A. Vandaele, M. Porcelli, and N. Gillis, *Accelerated Algorithms for Nonlinear Matrix Decomposition with the ReLU function*, arXiv Preprint [arXiv:2305.08687](https://arxiv.org/abs/2305.08687), 2023.
