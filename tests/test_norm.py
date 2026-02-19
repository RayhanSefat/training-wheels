from .adapters import run_layernorm, run_rmsnorm


def test_layernorm(numpy_snapshot, layernorm_input, layernorm_gamma, layernorm_beta):
    result = run_layernorm(layernorm_input, layernorm_gamma, layernorm_beta)
    numpy_snapshot.assert_match(result, atol=1e-6)


def test_rmsnorm(numpy_snapshot, rmsnorm_input, rmsnorm_gamma):
    result = run_rmsnorm(rmsnorm_input, rmsnorm_gamma)
    numpy_snapshot.assert_match(result, atol=1e-6)
