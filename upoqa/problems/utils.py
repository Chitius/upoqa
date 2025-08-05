# Copyright (c) 2025, Yichuan Liu and Yingzhou Li
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np


def rand_matrix_gen(n, min_eig, max_eig):
    A = np.diag(np.linspace(min_eig, max_eig, n))
    for _ in range(n):
        v = np.random.randn(n)
        v = v / np.linalg.norm(v) - np.eye(n)[:, 0]
        v = v / np.linalg.norm(v)
        H = np.eye(n) - 2 * np.outer(v, v)
        A = (H.T @ A) @ H
    return A


def smallest_eigenvectors(A, k):
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    indices = np.argsort(eigenvalues)[:k]
    smallest_eigenvalues = eigenvalues[indices]
    smallest_eigenvectors = eigenvectors[:, indices]
    return smallest_eigenvalues, smallest_eigenvectors.T
