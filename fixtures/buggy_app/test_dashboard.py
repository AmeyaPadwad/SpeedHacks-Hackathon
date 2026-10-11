from dashboard import render_dashboard


def test_known_user():
    assert render_dashboard(1) == "Sarah has 42 commits"


def test_unknown_user_does_not_crash():
    assert render_dashboard(2) == "Unknown user"
