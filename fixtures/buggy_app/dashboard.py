"""Demo fixture: a deliberate bug for the FlowState investigate -> coding agent flow."""


def load_user(user_id: int) -> dict | None:
    users = {1: {"name": "Sarah", "stats": {"commits": 42}}}
    return users.get(user_id)


def render_dashboard(user_id: int) -> str:
    user = load_user(user_id)
    # BUG: user can be None for unknown ids, and "stats" is accessed without a guard.
    return f"{user['name']} has {user['stats']['commits']} commits"


if __name__ == "__main__":
    print(render_dashboard(1))
    print(render_dashboard(2))
