function [g_, nablag_] = g(w_guess)
    global n_step ns t_init nu n_int int_step d_step s_fin s_init
    
    x_guess = w_guess(1:ns*(n_step+1));
    u_guess = [w_guess(ns*(n_step+1)+1:end); zeros(nu,1)];
    
    g_ = zeros(ns*(n_step+2),1);
    
    nablag_ = zeros(ns*(n_step+2), ns*(n_step+1) + nu*n_step);
       
    u_init = u_guess(1:nu);
    x_init = x_guess(1:ns);
    
    g_(1:ns) = x_init - s_init;
    nablag_(1:ns*(n_step+1),1:ns*(n_step+1)) = -eye(ns*(n_step+1));
    nablag_(1:ns,1:ns) = eye(ns); % first is positive
    % single shooting
    for k = 1:(n_step)

        tk = t_init + d_step*(k-1);
        [s,A,B] = expl_rk4(tk,x_init,u_init,eye(ns),zeros(ns,nu),int_step,n_int);
        
        u_init = u_guess(k*nu+1:(k+1)*nu);
        x_init = x_guess(k*ns+1:(k+1)*ns);
        
        g_(k*ns+1:(k+1)*ns) = s - x_init;
        nablag_(k*ns+1:(k+1)*ns, (k-1)*ns+1:k*ns) = A;
        nablag_(k*ns+1:(k+1)*ns, ns*(n_step+1)+1+(k-1)*nu:ns*(n_step+1)+(k-1)*nu +nu) = B;

    end
    
    % END CONSTRAINT
    k=k+1;
    g_(end-ns+1:end) = x_init - s_fin;
    nablag_(k*ns+1:(k+1)*ns, (k-1)*ns+1:k*ns) = eye(ns);
    
    nablag_ = nablag_.';
    
end
