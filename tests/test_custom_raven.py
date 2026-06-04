def test_custom_raven_uses_submodule_fla():
    from foreblocks.sequence.raven import Raven

    assert Raven.__module__ == "foreblocks.sequence.raven.blocks.raven"
