from collections import defaultdict, deque

MAX_HISTORY = 10

conversation_history = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY)
)


def get_conversation(key):
    return list(conversation_history[key])


def save_message(key, role, content):
    conversation_history[key].append({
        "role": role,
        "content": content
    })