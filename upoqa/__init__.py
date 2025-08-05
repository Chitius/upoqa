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

"""
UPOQA
=====

UPOQA is a derivative-free model-based optimization method for partially 
separable problem, whose name is the abbreviation for Unconstrained 
Partially-Separable Optimization By Quadratic Approximation.
"""

from .version import __version__
from . import utils
from .upoqa import minimize

__all__ = [
    '__version__',
    'minimize',
    'utils',
    'problems',
]
