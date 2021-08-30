function nablaLagrangian_sym = nablaLagrangian(x,lam,mu)
    [~,nablag_] = g(x);
    nablaLagrangian_sym = nablaf(x) + nablag_*lam + nablah(x)*mu;
end