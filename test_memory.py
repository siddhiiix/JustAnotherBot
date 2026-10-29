from ai.conversation import get_conversation, save_message, DB_PATH

# Use a test conversation so we don't modify your Discord history.
test_key = (999, 888, 777)

save_message(test_key, "user", "My favourite colour is purple.")

print("Database:", DB_PATH)
print("\nRetrieved conversation:")

for message in get_conversation(test_key):
    print(message)
