# Domain truncated Local Natural Orbital methods

![Build Status](https://github.com/fishjojo/dlno/actions/workflows/ci.yml/badge.svg?branch=master)
[![codecov](https://codecov.io/github/fishjojo/dlno/graph/badge.svg?token=Z0HUZOXFO7)](https://codecov.io/github/fishjojo/dlno)

The `dlno` package implements the (spin-restricted) local MP2 method,
which largely follows the implementation of
DOI: [10.1021/acs.jctc.6b00732](https://doi.org/10.1021/acs.jctc.6b00732).
It is also used as a domain builder for performing
local natural orbital coupled cluster (LNO-CC) calculations within the domains
to achieve better performance.
The key workflow is summarized as follows.

* Full-system RHF calculation.

* Localize the active occupied orbitals (LMOs). 
`DLNO.build_lmo`: Pipek–Mezey (default), Boys or ER;
relevant parameter `DLNO.lmo_method` (default `pm`).
The LMOs are saved as `DLNO.lmo`.

* For each LMO ($\psi_{i}$), get its Boughton-Pulay (BP; DOI: [10.1002/jcc.540140615](https://doi.org/10.1002/jcc.540140615)) atom domain,
which corresponds to a set of atoms whose AOs ($\chi$) can best span $\psi_{i}$ up to certain accuracy,
with the least-squares residual defined as
$$f(\mathbf{a}_i) = \min_{\mathbf{a}_i}\left[\int (\psi_{i}(\mathbf{r}) - \sum_{\mu \in D_\text{BP}^\text{lmo}(i)} a_{i \mu} \chi_{\mu}(\mathbf{r}))^2 d\mathbf{r}\right] = 1 - \sigma_{\text{BP}}^{\text{lmo}}$$
where $\sigma_\text{BP}^\text{lmo}$ is the parameter `DLNO.lmo_bp_domain_thr` (default `0.999`)
The BP domain for LMOs are saved as `DLNO.lmo_bp_domain`.

* Get PAOs (AOs projected onto the active virtual space), saved as `DLNO.pao` along with the AO&rarr;PAO mapping `DLNO.ao2pao_map` [PAOs with norm smaller than `DLNO.pao_norm_thr` (default `1e-4`) are discarded].

* For each PAO, get its BP domain, saved as `DLNO.pao_bp_domain`. $\sigma_\text{BP}^\text{pao}$ is set by the parameter `DLNO.pao_bp_domain_thr` (default `0.98`).

* For each LMO, get its **primary domain**, defined as the union of its BP domain and the BP domains of all PAOs centered on the atoms belonging to that LMO's BP domain. It is saved as `DLNO.lmo_primary_domain`.

* For each LMO, PAOs centered on its BP domain atoms are canonically orthogonalized, and are then truncated according to the domain projected overlap matrix
$$\tilde{S}_{ab}(i) = \langle a | \left( \sum_{\mu\nu \in D_\text{BP}^\text{lmo}(i)} |\mu \rangle S_{\mu \nu}^{-1} \langle \nu| \right) | b\rangle$$
$\tilde{S}$ is diagonalized and its eigenvectors with eigenvalues greater than `DLNO.domain_pao_thr` (default `1e-4`) are selected as the **domain projected PAOs (DPAOs)**. The relevant function is `DLNO.build_domain_pao`.

* For each LMO and its DPAOs, project them onto the primary domain space and Löwdin orthonormalize. The resulting orbitals are semi-canonicalized for use in the following pair energy calculation.

* The MP2 pair energy is approximated with multipole expansion up to 4th order (`DLNO.multipole_order=4`):
$$E_{ij} = 2\sum_{ab} \frac{(ai|bj) (ia|jb)}{e_i + e_j - e_a - e_b}$$
The exchange term is dropped as it decays to zero for distant pairs.
The relevent function is `dlno.mp2.multipole_order`.
>**Note:** In the code, $E_{ij}$ is multiplied by 2, because the total energy is computed by summing over $i$ and $j$, and $E_{ij}$ is used to determine the strong pairs to keep for local MP2 calculations ($|E_{ij}|$ > `DLNO.pair_energy_thr`; default `1e-4`).

* For each LMO, its **extended domain** is defined as the union of the BP domains of its strong pair partners (including itself). Similarly, the **extended primary domain (EP domain)** is the union of their primary domains.

* LMOs are grouped by identical EP domain.
For each unique EP domain, density fitting integrals are computed. All the following calculations will be performed within the EP domain AO space.

* The LMOs with the same EP domain is further grouped by identical extended domain. 

* For the set of LMOs with both the same extended domain and EP domain, DPAOs are generated for the extended domain and projected onto the EP domain.

* The LMOs are further grouped by identical strong pairs, and they are chosen as the central LMOs. The central LMOs along with their strong pairs are projected onto the EP domain, Löwdin orthonormalized, and semi-canonicalized. These form the occupied space for the local MP2 calculation.

* The DPAOs are orthogonalized against the occupied space above, Löwdin orthonormalized, and semi-canonicalized. These form the virtual space for the local MP2 calculation.

* Local MP2 calculation is then performed within this local occupied plus virtual space. Optional LNO-CC calculations can also be performed.

* Total correlation energy is obtained by summing over local correlation energy for each set of central LMOs, and finally by adding the distant pair energies.
