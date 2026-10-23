import numpy as np
import pytest

from scripts.Hamiltonian import Hamiltonian
from scripts.Lattice import Lattice
from scripts.partial_sc import Corr, partial_sc
from scripts.self_consistency import bdg_sc

ABS_TOL = 1e-3
REL_TOL = 0.1
TEMPERATURE = 0.1


def make_hamiltonian(*, x=4, y=4, hopping=0.0, onsite=4.0,
                     chemical_potential=0.0, initial_onsite=0.25):
    return Hamiltonian(
        t=hopping,
        mu=np.full(x, chemical_potential, dtype=float),
        lattice=Lattice(X=x, Y=y),
        U=np.full(x, onsite, dtype=float),
        V=np.zeros(x, dtype=float),
        V_prime=np.zeros(x, dtype=float),
        F0_init=initial_onsite,
    )


def correlations(H):
    """The solvers mutate H; compare all 13 channels, including triplets."""
    return {field: np.asarray(getattr(H, field)).copy() for field in Corr._fields}


def assert_correlations_close(actual, expected):
    assert actual.keys() == expected.keys()
    for field in actual:
        np.testing.assert_allclose(
            actual[field], expected[field], atol=ABS_TOL, rtol=REL_TOL,
            err_msg=f"Disagreement in {field}",
        )


def run_full(H, *, maxiter=30, mixing=1.0):
    bdg_sc(H, TEMPERATURE, atol=1e-6, rtol=1e-4,
           maxiter=maxiter, mixing=mixing)
    return H


def run_partial(H, *, fixed_sites=None, fixed_syms=None,
                maxiter=30, mixing=1.0):
    partial_sc(H, TEMPERATURE, fixed_sites=fixed_sites,
               fixed_syms=fixed_syms, atol=1e-6, rtol=1e-4,
               maxiter=maxiter, mixing=mixing)
    return H


@pytest.mark.parametrize("solve", [run_full, run_partial], ids=["fully-self-consistent", "partial-unfixed"])
def test_uniform_bulk_atomic_limit_has_expected_onsite_amplitude(solve):
    H = make_hamiltonian(x=4, y=4, hopping=0.0, onsite=4.0)
    solve(H)
    assert H.converged, "The uniform atomic-limit calculation did not converge"
    np.testing.assert_allclose(
        H.F0, np.full(4, 0.625), atol=ABS_TOL, rtol=REL_TOL,
    )
    np.testing.assert_allclose(
        H.F0, np.full(4, H.F0.mean()), atol=ABS_TOL, rtol=REL_TOL,
    )
    # No hopping or triplet interactions: all x bonds and triplet channels vanish.
    for field in ("F_xplus", "F_xmin", *Corr._fields[5:]):
        np.testing.assert_allclose(
            getattr(H, field), 0.0, atol=ABS_TOL, rtol=REL_TOL,
            err_msg=f"Unexpected pairing in {field}",
        )
    assert np.isfinite(H.free)


@pytest.mark.parametrize("solve", [run_full, run_partial], ids=["fully-self-consistent", "partial-unfixed"])
def test_normal_state_has_zero_pairing_correlations(solve):
    """With no attractive interactions or seed, all pairing must vanish."""
    H = make_hamiltonian(
        x=4, y=4, hopping=0.7, onsite=0.0,
        chemical_potential=0.83, initial_onsite=0.0,
    )
    solve(H)
    assert H.converged
    for field, values in correlations(H).items():
        np.testing.assert_allclose(
            values, np.zeros_like(values), atol=ABS_TOL, rtol=REL_TOL,
            err_msg=f"Normal-state {field} should be zero",
        )


def test_partial_with_no_fixed_sites_matches_one_full_iteration():
    """An unfixed partial iteration must implement the full update map."""
    kwargs = dict(x=4, y=4, hopping=0.4, onsite=3.0,
                  chemical_potential=0.2, initial_onsite=0.25)
    full = run_full(make_hamiltonian(**kwargs), maxiter=1, mixing=0.6)
    partial = run_partial(
        make_hamiltonian(**kwargs), fixed_sites=np.zeros(4, dtype=bool),
        fixed_syms=("F0", "F_yplus"), maxiter=1, mixing=0.6,
    )
    assert_correlations_close(correlations(partial), correlations(full))
    np.testing.assert_allclose(
        partial.free, full.free, atol=ABS_TOL, rtol=REL_TOL,
    )


def test_partial_keeps_selected_correlations_fixed_and_updates_others():
    """Fixed positions of named channels remain unchanged (not zeroed)."""
    kwargs = dict(x=4, y=4, hopping=0.4, onsite=3.0,
                  chemical_potential=0.2, initial_onsite=0.25)
    mask = np.array([True, False, True, False])
    initial = make_hamiltonian(**kwargs)
    before = correlations(initial)

    partial = run_partial(
        initial, fixed_sites=mask, fixed_syms=("F0", "F_yplus"),
        maxiter=1, mixing=1.0,
    )
    full = run_full(make_hamiltonian(**kwargs), maxiter=1, mixing=1.0)
    observed = correlations(partial)
    reference = correlations(full)

    for field in ("F0", "F_yplus"):
        np.testing.assert_array_equal(observed[field][mask], before[field][mask])
        np.testing.assert_allclose(
            observed[field][~mask], reference[field][~mask],
            atol=ABS_TOL, rtol=REL_TOL,
        )

    # Channels that were NOT named as fixed should be updated everywhere.
    for field in Corr._fields:
        if field not in ("F0", "F_yplus"):
            np.testing.assert_allclose(
                observed[field], reference[field], atol=ABS_TOL, rtol=REL_TOL,
                err_msg=f"Unfixed channel {field} did not follow full solver",
            )
    assert np.max(np.abs(observed["F0"][~mask] - before["F0"][~mask])) > ABS_TOL


def test_partial_rejects_unknown_correlation_name():
    with pytest.raises(ValueError, match="Invalid correlation name"):
        run_partial(
            make_hamiltonian(), fixed_sites=np.ones(4, dtype=bool),
            fixed_syms=("not_a_correlation",), maxiter=1,
        )


@pytest.mark.parametrize("mixing", [0.0, -0.2, 1.01])
def test_partial_rejects_invalid_mixing(mixing):
    with pytest.raises(ValueError, match="mixing"):
        run_partial(make_hamiltonian(), mixing=mixing, maxiter=1)
