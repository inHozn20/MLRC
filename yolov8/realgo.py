import asyncio
from bleak import BleakClient, BleakScanner

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHAR_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
TARGET_NAME = "ESP32_BLE_Server"



async def main():
    # BLE 장치 스캔
    print("Scanning for ESP32...")
    devices = await BleakScanner.discover()
    target = None
    for d in devices:
        if d.name == TARGET_NAME:
            target = d
            break

    if not target:
        print("ESP32 not found.")
        return

    print(f"Connecting to {target.address}...")

    async with BleakClient(target.address) as client:
        if not client.is_connected:
            print("Failed to connect.")
            return

        print("Connected.")
        while True:
            msg = input("Send to ESP32 (type 'exit' to quit): ")
            if msg.lower() == 'exit':
                break
            await client.write_gatt_char(CHAR_UUID, msg.encode())



asyncio.run(main())
