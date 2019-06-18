def test_version():
    import mygrad

    assert isinstance(mygrad.__version__, str)
    assert mygrad.__version__
    assert "unknown" not in mygrad.__version__
