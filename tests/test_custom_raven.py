def test_custom_raven_uses_submodule_fla():
    from foreblocks.models.sequence.raven import Raven

    assert Raven.__module__ == "foreblocks.models.sequence.raven.blocks.raven"
