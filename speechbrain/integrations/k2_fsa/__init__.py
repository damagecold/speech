"""
Package providing `k2-fsa <https://github.com/k2-fsa/k2>`_ integration.

Intended loading manner:

    >>> import speechbrain.integrations.k2_fsa as sbk2
    >>> # Then use: sbk2.graph_compiler.CtcGraphCompiler for example

"""

try:
    import k2  # noqa
except ImportError:
    # k2 not available, skipping
    k2 = None

from speechbrain.utils.importutils import lazy_export_all

lazy_export_all(__file__, __name__, export_subpackages=True)
