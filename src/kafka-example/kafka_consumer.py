from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "mytopic03",
    bootstrap_servers=["localhost"],
    auto_offset_reset="earliest",
    enable_auto_commit=False,
    group_id="1",
)

# a = input()
for message in consumer:
    print(message)
    print(message.value)