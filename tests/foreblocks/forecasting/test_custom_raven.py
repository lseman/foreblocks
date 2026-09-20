def test_custom_raven_uses_submodule_fla():
    from foreblocks.forecasting.sequence.raven import Raven

    assert Raven.__module__ == "foreblocks.forecasting.sequence.raven.blocks.raven"
