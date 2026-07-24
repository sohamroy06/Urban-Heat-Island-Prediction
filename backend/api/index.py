from main import app, app_state, load_or_train


if app_state["models"] is None:
    load_or_train()
