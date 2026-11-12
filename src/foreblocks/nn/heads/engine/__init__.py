"""Head composition and execution.

``composer`` owns HeadComposer configuration, registered state, serial forward
execution, and the public interface. Private stateless mixins in ``parallel``
and ``inverse`` supply parallel fusion and serial inverse execution. All
parameters and lazy projections stay registered on the composer.

``alignment``, ``execution``, and ``graph`` support stage configuration and
composition through HeadGraph.
"""
