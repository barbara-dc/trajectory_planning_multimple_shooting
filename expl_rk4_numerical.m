function [x_plus,A,B] = expl_rk4_numerical(t0,x,u,A,B,h,n_int)
    [x_plus,~,~] = expl_rk4(t0,x,u,A,B,h,n_int);
    epsilon = sqrt(h);
    nx = length(x);
    nu = length(u);
    A = zeros(nx);
    B = zeros(nx, nu);
    E = eye(nx);
    Eu = eye(nu);
    for i=1:nx
        x_ = x + epsilon*E(:,i);
        [x_plus_,~,~] = expl_rk4(t0,x_,u,A,B,h,n_int);
        A(:,i) = (x_plus_ - x_plus) / epsilon;
    end
    for i=1:nu
        u_ = u + epsilon*Eu(:,i);
        [x_plus_,~,~] = expl_rk4(t0,x,u_,A,B,h,n_int);
        B(:,i) = (x_plus_ - x_plus) / epsilon;
    end
end