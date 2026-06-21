import json


def save_history_to_json(history, path):
    serializable_data = {
        key: [float(x) for x in values] for key, values in history.items()
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2)