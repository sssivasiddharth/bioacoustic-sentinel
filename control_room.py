# control_room.py
# Receiver script – simulates a ranger station getting real‑time alerts
import socket

HOST, PORT = 'localhost', 5000

print("🟢 Control room listening for alerts…\n")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(f"\n🚨 RECEIVED: {data.decode().strip()}\n")