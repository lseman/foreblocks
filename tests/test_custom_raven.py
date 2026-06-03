def test_custom_raven_uses_submodule_fla():
    from foreblocks.custom_raven import Raven

    assert Raven.__module__ == "fla.layers.raven"
