API Reference
=============

.. automodule:: array_api_strict

Array API Strict Flags
----------------------

.. automodule:: array_api_strict._flags

.. currentmodule:: array_api_strict

.. autofunction:: get_array_api_strict_flags
.. autofunction:: set_array_api_strict_flags
.. autofunction:: reset_array_api_strict_flags
.. autoclass:: ArrayAPIStrictFlags

.. _environment-variables:

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Flags can also be set with environment variables.
:func:`set_array_api_strict_flags` will override the values set by environment
variables. Note that the environment variables will only change the defaults
used by array-api-strict initially. They will not change the defaults used by
:func:`reset_array_api_strict_flags`.

.. envvar:: ARRAY_API_STRICT_API_VERSION

   A string representing the version number.

.. envvar:: ARRAY_API_STRICT_DATA_DEPENDENT_SHAPES

   "True" or "False" to enable or disable data dependent shapes.

.. envvar:: ARRAY_API_STRICT_ENABLED_EXTENSIONS

   A comma separated list of extensions to enable.