DEBUG = False
n_jobs = -1
DIFFERENTIAL_EVOLUTION_KWARGS = dict(
    workers=n_jobs,
    maxiter=50,
    polish=False,
    updating='deferred' if n_jobs == -1 else 'immediate'
)
