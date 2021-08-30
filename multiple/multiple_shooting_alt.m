function [R, nablaR] = multiple_shooting(w_guess, ~)
    global n_step ns nu  state_coeff input_coeff s_fin 
    state_coeff_sqrt = sqrt(state_coeff);
    input_coeff_sqrt = sqrt(input_coeff);
    
    w_des = [kron(ones(n_step+1,1), s_fin) ; zeros(n_step*nu, 1)];

    
    nablaR = [kron(eye(n_step+1),state_coeff_sqrt) zeros(ns*(n_step+1), nu*n_step);
        zeros(nu*n_step, ns*(n_step+1)) kron(eye(n_step),input_coeff_sqrt)];
    R = nablaR*(w_guess-w_des);


end