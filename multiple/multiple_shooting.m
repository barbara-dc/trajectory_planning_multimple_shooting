function [R, nablaR] = multiple_shooting(w_guess, ~)

    global n_step ns nu  state_coeff input_coeff dijkstra_path

    state_coeff_sqrt = sqrt(state_coeff);
    input_coeff_sqrt = sqrt(input_coeff);
    w_des = dijkstra_path(1,:).';
    for i=1:n_step
        w_des = [w_des; zeros(nu, 1); dijkstra_path(i+1,:).'];
    end
    

    coeff_sqrt = [input_coeff_sqrt zeros(nu, ns);
        zeros(ns, nu) state_coeff_sqrt];
    nablaR = [state_coeff_sqrt zeros(ns, (ns+nu)*n_step);
        zeros((ns+nu)*n_step, ns) kron(eye(n_step),coeff_sqrt)];

    R = nablaR*(w_guess-w_des);

end