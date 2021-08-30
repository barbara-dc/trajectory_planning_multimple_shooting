function [g_, nablag_] = g(w_guess)
    global n_step ns t_init nu n_int int_step d_step s_fin s_init
    
    x_guess = w_guess(1:ns);
    for i = 1:n_step
        x_guess = [x_guess; w_guess(i*(ns+nu)+1:i*(ns+nu)+ns)];
    end
    u_guess = w_guess(ns+1:ns+nu);
    for i = 1:n_step-1
        u_guess = [u_guess; w_guess(i*(ns+nu)+ns+1:i*(ns+nu)+ns+nu)];
    end
    
    g_ = zeros(ns*(n_step+2),1);
    
    nablag_ = zeros(ns*(n_step+2), ns*(n_step+1) + nu*n_step);
    
    x_init = x_guess(1:ns);
    
    g_(1:ns) = x_init - s_init;
    dxdw = zeros(ns, ns*(n_step+1) + nu*n_step);
    dxdw(:,1:ns) = eye(ns);
    nablag_(1:ns,:) = dxdw;
    for k = 1:(n_step)

        tk = t_init + d_step*(k-1);
        u_init = u_guess((k-1)*nu+1:k*nu);
        [s,A,B] = expl_rk4(tk,x_init,u_init,eye(ns),zeros(ns,nu),int_step,n_int);
        
        x_init = x_guess(k*ns+1:(k+1)*ns);
        
        g_(k*ns+1:(k+1)*ns) = s - x_init;
        dxdw = [zeros(ns,(k-1)*ns+(k-1)*nu) A B -eye(ns) zeros(ns,(n_step-k)*ns+(n_step-k)*nu)];
        nablag_(k*ns+1:(k+1)*ns,:) = dxdw;

    end
    
    % END CONSTRAINT
    k=k+1;
    dxdw = 0*dxdw;
    g_(end-ns+1:end) = x_init - s_fin;
    dxdw(:,end-ns+1:end) = eye(ns);
    nablag_(k*ns+1:(k+1)*ns, :) = dxdw;
    
    nablag_ = nablag_.';
    
end
